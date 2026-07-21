# chito

`chito` is a small GRPO rollout service for distributed training. Training code
constructs one `RolloutEngine`; rank 0 then starts a Ray actor that owns the
dataset, Workflow, and vLLM. The training processes only request batches and
push updated inference weights.

V1 supports:

- Python 3.11 or newer;
- vLLM 0.25 and Ray on one server;
- synchronous rollout or one-batch asynchronous lookahead;
- NCCL weight transfer from training rank 0 to vLLM;
- ZeRO-2/DDP-style training where rank 0 has the complete synchronized model.

SGLang, per-rank streaming, ZeRO-3/FSDP sharded weight transfer, and partial
agent rollout resume are not implemented in V1.

## Installation

Install the vLLM runtime and training dependencies:

```bash
python -m pip install -e '.[vllm,train]'
```

The declared vLLM range is `>=0.25,<0.26`. PyTorch, CUDA, and vLLM must use
compatible builds for the local NVIDIA driver.

## Public API

The Engine has three constructor arguments and three public async methods:

```python
from chito import RolloutEngine


engine = RolloutEngine(dataset=dataset, workflow=workflow, config=config)

batch = await engine.next_batch()
new_policy_version = await engine.update_weights(model)  # rank 0 only
await engine.aclose()
```

`RolloutConfig` is flat. Only four fields are required:

```python
from chito import RolloutConfig

config = RolloutConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    group_size=8,
    per_device_train_batch_size=32,
    max_concurrent_groups=64,
    rollout_mode="async",       # "sync" or "async"
    rollout_gpu_ids=(0,),        # physical GPU IDs used by vLLM
    max_tokens=1024,
    temperature=1.0,
    backend_kwargs={
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
    },
)
```

The global number of samples in one training step is
`per_device_train_batch_size * train_world_size`. It must be divisible by
`group_size`.

## Rollout semantics

Each dataset item becomes one complete GRPO group:

```text
dataset item
  -> Workflow.prepare
  -> group_size concurrent Workflow.run calls
  -> Workflow.reward for every sample
  -> Workflow.postprocess
  -> Workflow.compute_advantages on the complete group
  -> independent TrainingSample values
  -> fixed contiguous shard for each training rank
```

Advantages are therefore computed before a group may cross a rank boundary. A
Workflow may return `None` from `postprocess` to reject a whole group; the
service keeps reading dataset items until the batch is full.

Both rollout modes return the same `TrainingBatch` shape:

- `sync` generates the next batch only after the preceding weight update has
  completed.
- `async` trains batch N while vLLM prepares batch N+1 with the current
  inference weights. `update_weights()` waits for that one prefetched batch,
  updates vLLM synchronously, and then releases it. After warmup, its behavior
  policy is one version behind the learner.

V1 never queues a third batch and does not release one rank early. Every rank's
`next_batch()` waits for the complete global batch and then receives its fixed
local shard.

## Workflow

For ordinary single-turn GRPO, subclass `GRPOWorkflow` and implement only prompt
preparation and reward calculation:

```python
from chito import GRPOWorkflow, RolloutPrompt


class MathWorkflow(GRPOWorkflow):
    async def setup(self) -> None:
        # Load the tokenizer inside the rollout actor.
        ...

    async def prepare(self, item: object, item_id: str) -> RolloutPrompt:
        token_ids = ...
        return RolloutPrompt(item_id, tuple(token_ids), {"answer": item["answer"]})

    async def reward(self, prompt, sample) -> float:
        return ...
```

`run`, `postprocess`, and `compute_advantages` may also be overridden. A custom
`run` can perform multiple model/tool/environment turns, but V1 waits for the
entire run before changing inference weights; it does not pause and resume a
live trajectory across policy versions.

## Distributed training

Constructing `RolloutEngine` is collective across the initialized training
process group. Rank 0 supplies the dataset and Workflow; the other ranks pass
`None` and connect to the same rollout actor:

```python
dataset = load_training_data() if accelerator.is_main_process else None
workflow = MathWorkflow() if accelerator.is_main_process else None
engine = RolloutEngine(dataset, workflow, config)

try:
    for step in range(num_steps):
        batch = await engine.next_batch()  # called by every rank

        loss = compute_loss(model, batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step + 1 < num_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                await engine.update_weights(accelerator.unwrap_model(model))
finally:
    await engine.aclose()  # called by every rank
```

DeepSpeed, DDP, Accelerate, or `torchrun` remains responsible for synchronizing
the training ranks. `update_weights()` does not run a training barrier or copy
parameters between training ranks; it only sends rank 0's already synchronized
complete CUDA parameters to vLLM. V1 therefore targets ZeRO-2 rather than
sharded-parameter ZeRO-3/FSDP.

## GSM8K example

[`samples/async_gsm8k_grpo.py`](samples/async_gsm8k_grpo.py) contains a complete,
flat Accelerate + DeepSpeed training loop with format and answer rewards. For
example, reserve physical GPU 0 for vLLM and train on GPUs 1 through 4:

```bash
CHITO_MODEL=/models/Qwen2.5-0.5B-Instruct \
CHITO_ROLLOUT_MODE=async \
CHITO_ROLLOUT_GPU_IDS=0 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
accelerate launch --use_deepspeed --num_processes 4 \
  samples/async_gsm8k_grpo.py
```

Set `CHITO_GSM8K_TRAIN` to a local GSM8K JSONL file when training offline. Set
`CHITO_ROLLOUT_MODE=sync` to run the matched synchronous schedule without
changing the training loss.

## Future work

The exact V1 double-buffer protocol and the proposed V2/V3 boundaries are in
[`async_plan.md`](async_plan.md). The comparison with verl and slime is in
[`research.md`](research.md). Those plans do not imply that streaming rollout,
partial resume, SGLang, or multi-node recovery already exists.
