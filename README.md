# chito

`chito` is a small, backend-agnostic asynchronous rollout coordinator for GRPO.
It overlaps complete-group rollout production with training-batch consumption and
switches inference weights only at group boundaries.

V1 requires Python 3.11 or newer. Its optional `VllmBackend` implements the
`InferenceBackend` boundary with vLLM's asynchronous Python engine. Other
inference systems can implement the same small protocol without changing the
rollout engine.

## Core semantics

- `group_size` is fixed, must be at least two, and belongs to the engine rather
  than the workflow.
- `train_batch_size` counts complete prompt groups, not individual samples.
- Groups run concurrently up to `max_concurrent_groups`. Samples within a group
  also run concurrently.
- Every complete group uses one immutable `policy_version`.
- The accepted-group buffer is bounded at
  `train_batch_size * prefetch_batches`.
- A group post-hook can replace a complete group or return `None` to reject it.
  Replacement groups are checked for prompt identity, sample count, sample
  indices, rewards, and policy-version consistency.
- `update_weights()` closes new admission, waits for all active groups to finish,
  updates the backend, increments the version, and then reopens admission.

`rollout(source)` registers exactly one asynchronous prompt source, starts an
owned background producer, and returns immediately. Producer failures are raised
from `next_batch()` as `RolloutFailedError`.

## Backend and workflow contracts

```python
from chito import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    RolloutContext,
    RolloutPrompt,
    RolloutSample,
    RolloutWorkflow,
)


class MyBackend(InferenceBackend):
    async def generate(self, request: InferenceRequest) -> InferenceResult:
        ...

    async def update_weights(
        self,
        update: object,
        *,
        new_policy_version: int,
    ) -> None:
        ...

    async def aclose(self) -> None:
        ...


class MyWorkflow(RolloutWorkflow):
    async def run(
        self,
        context: RolloutContext,
        prompt: RolloutPrompt,
    ) -> RolloutSample:
        ...
```

A workflow produces one unrewarded sample. The engine invokes the asynchronous
reward function, forms the fixed-size group, and then invokes the optional group
post-hook. `SingleTurnWorkflow` implements the common one-generation workflow.

Final `RolloutSample` values preserve the exact full token sequence, behavior
log-probabilities, loss mask, reward, and policy version. Prompt and non-policy
context tokens may use `loss_mask=False`.

## vLLM backend

Install the optional dependency, or install `chito` into an environment that
already contains a compatible vLLM build:

```bash
python -m pip install -e '.[vllm]'
```

CUDA wheels must match the installed NVIDIA driver. For example, a separate
CUDA 12.9 environment can be created with `uv` without embedding any
machine-specific path in the project:

```bash
VENV_PATH="${VENV_PATH:-.venv-vllm}"
uv venv --python 3.12 "$VENV_PATH"
uv pip install --python "$VENV_PATH/bin/python" \
  vllm==0.24.0 --torch-backend=cu129
uv pip install --python "$VENV_PATH/bin/python" -e '.[test]'
```

The adapter targets vLLM 0.24's asynchronous engine and public checkpoint reload
API. The optional dependency is bounded below by 0.24 and below the next
unverified 0.25 release.

Constructing `VllmBackend` loads the model. Generation passes the exact prompt
token IDs to vLLM, requests the sampled-token logprob at every generated
position, and lets vLLM schedule concurrent asynchronous requests:

```python
from chito import VllmBackend

backend = VllmBackend(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens=64,
    temperature=0.8,
    engine_kwargs={
        "dtype": "float16",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
    },
)
```

`engine_kwargs` are forwarded to vLLM's public `AsyncEngineArgs`. The adapter
owns the loaded engine; call `aclose()` directly, or let `RolloutEngine.aclose()`
close it.

V1 supports synchronous updates from a complete local Hugging Face checkpoint.
Save each trained policy to a new immutable directory before submitting it:

```python
from chito import VllmWeightUpdate

update = VllmWeightUpdate(
    "/checkpoints/policy-0001",
)
new_policy_version = await engine.update_weights(update)
```

The backend blocks new generation and drains its own active requests, pauses
vLLM with cache clearing, invokes vLLM's public `reload_weights` collective RPC,
and resumes generation only after every worker has loaded the complete
checkpoint. The checkpoint path is resolved to an absolute path and must be
visible to every vLLM worker.

An invalid or deleted checkpoint directory is rejected before vLLM is paused and
can be corrected and retried. Once pause, reload, or resume fails, the loaded
model may contain partial weights; the backend becomes poisoned and rejects
generation and further updates. Close it and load a fresh backend rather than
assigning a policy version to uncertain weights.

## Usage

```python
from collections.abc import AsyncIterator

from chito import (
    RolloutConfig,
    RolloutEngine,
    RolloutPrompt,
    SingleTurnWorkflow,
    SourceExhaustedError,
)


async def prompt_source() -> AsyncIterator[RolloutPrompt]:
    yield RolloutPrompt(
        prompt_id="task-1",
        token_ids=(101, 102),
        metadata={"answer": 42},
    )


async def reward(prompt, sample) -> float:
    return score(sample, expected=prompt.metadata["answer"])


engine = RolloutEngine(
    backend=backend,
    workflow=SingleTurnWorkflow(),
    reward_function=reward,
    config=RolloutConfig(
        group_size=8,
        train_batch_size=32,
        max_concurrent_groups=64,
        prefetch_batches=1,
    ),
    post_hook=optional_group_filter,
)

await engine.rollout(prompt_source())
try:
    while True:
        batch = await engine.next_batch()
        update = await trainer.train(batch)
        new_version = await engine.update_weights(update)
except SourceExhaustedError as exhausted:
    # next_batch never returns a partial batch. Any final accepted groups are
    # exposed on the exception instead of being silently discarded.
    final_groups = exhausted.remaining_groups
finally:
    await engine.aclose()
```

If the source or any rollout task fails, `next_batch()` fails immediately rather
than waiting forever. Closing the engine cancels internal work, wakes blocked
producers and consumers, and closes the backend exactly once.

## Samples

[`samples/train_grpo.py`](samples/train_grpo.py) is a minimal educational
clipped-GRPO loop using the current synchronous checkpoint reload flow:

```bash
python -m pip install -e '.[vllm]'
python samples/train_grpo.py
```

On smaller GPUs, lower the sample's `gpu_memory_utilization` setting.

## Tests

The test suite uses a deterministic fake backend and standard `asyncio.run`, so no
async pytest plugin is required:

```bash
python -m pytest
```

The real vLLM integration test is opt-in and is skipped by the command above. It
loads `Qwen/Qwen2.5-0.5B-Instruct`, verifies exact rollout token IDs and
logprobs, reloads a zeroed trainer checkpoint from disk, and verifies the next
policy version uses the updated weights:

```bash
CHITO_RUN_VLLM_INTEGRATION=1 \
  python -m pytest tests/integration/test_vllm_smoke.py -s
```

The test requires a working CUDA/vLLM environment and either network access or
a populated model cache. `CHITO_VLLM_MODEL` can select another local path or
model ID. Memory-related defaults can be overridden with
`CHITO_VLLM_GPU_MEMORY_UTILIZATION` and `CHITO_VLLM_MAX_MODEL_LEN`.
