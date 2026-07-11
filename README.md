# chito

`chito` is a small, backend-agnostic asynchronous rollout coordinator for GRPO.
It overlaps complete-group rollout production with training-batch consumption and
switches inference weights only at group boundaries.

V1 requires Python 3.11 or newer. It intentionally contains no concrete vLLM or
SGLang transport; those systems are integrated through `InferenceBackend`.

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

## Tests

The test suite uses a deterministic fake backend and standard `asyncio.run`, so no
async pytest plugin is required:

```bash
python -m pytest
```
