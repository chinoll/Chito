"""Minimal end-to-end GRPO training loop with vLLM rollouts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chito import (
    RolloutConfig,
    RolloutEngine,
    RolloutPrompt,
    RolloutSample,
    SingleTurnWorkflow,
    TrainingBatch,
    VllmBackend,
    VllmCheckpointWeightUpdate,
)


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
DTYPE = torch.float16
GROUP_SIZE = 4
MAX_NEW_TOKENS = 24
LEARNING_RATE = 1e-5
CLIP_EPSILON = 0.2
CHECKPOINT_ROOT = Path("checkpoints/grpo")
TOY_TASKS = (
    ("Name the color of a clear daytime sky in one short sentence.", "blue"),
    ("Write two plus two in words in one short sentence.", "four"),
    ("Name the opposite of cold in one short sentence.", "warm"),
)


def clipped_grpo_loss(
    model: torch.nn.Module,
    batch: TrainingBatch,
) -> torch.Tensor:
    sample_losses: list[torch.Tensor] = []

    for group in batch.groups:
        rewards = torch.tensor(
            [sample.reward for sample in group.samples],
            dtype=torch.float32,
            device=DEVICE,
        )
        advantages = (rewards - rewards.mean()) / (
            rewards.std(unbiased=False) + 1e-6
        )

        for sample, advantage in zip(group.samples, advantages, strict=True):
            token_ids = torch.tensor(sample.token_ids, device=DEVICE)
            logits = model(token_ids[:-1].unsqueeze(0)).logits[0]

            # Logit t predicts token t+1, so old and current logprobs must both
            # align with token_ids[1:]. The shifted mask keeps generated tokens only.
            next_tokens = token_ids[1:]
            current_logprobs = torch.log_softmax(logits.float(), dim=-1)
            current_logprobs = current_logprobs.gather(
                -1, next_tokens.unsqueeze(-1)
            ).squeeze(-1)
            behavior_logprobs = torch.tensor(
                sample.logprobs[1:], dtype=torch.float32, device=DEVICE
            )
            generated_mask = torch.tensor(
                sample.loss_mask[1:], dtype=torch.bool, device=DEVICE
            )

            ratio = torch.exp(current_logprobs - behavior_logprobs)
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON
            ) * advantage
            sample_losses.append(
                -torch.minimum(unclipped, clipped)[generated_mask].mean()
            )

    return torch.stack(sample_losses).mean()


async def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    trainer = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=DTYPE,
    ).to(DEVICE)
    trainer.config.use_cache = False
    trainer.train()
    optimizer = torch.optim.SGD(trainer.parameters(), lr=LEARNING_RATE)

    prompts: list[RolloutPrompt] = []
    for index, (question, target) in enumerate(TOY_TASKS):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompts.append(
            RolloutPrompt(
                prompt_id=f"toy-{index}",
                token_ids=tuple(token_ids),
                metadata={"target": target},
            )
        )

    release_next_prompt = [asyncio.Event() for _ in prompts[:-1]]

    async def prompt_source():
        for index, prompt in enumerate(prompts):
            if index > 0:
                await release_next_prompt[index - 1].wait()
            yield prompt

    async def reward(prompt: RolloutPrompt, sample: RolloutSample) -> float:
        output_ids = sample.token_ids[len(prompt.token_ids) :]
        answer = tokenizer.decode(output_ids, skip_special_tokens=True).lower()
        target = str(prompt.metadata["target"]).lower()
        return float(target in answer) - 0.01 * len(output_ids)

    run_directory = CHECKPOINT_ROOT / uuid4().hex
    run_directory.mkdir(parents=True)
    backend = VllmBackend(
        MODEL,
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=1.0,
        weight_transfer="checkpoint",
        engine_kwargs={
            "dtype": "float16",
            "enforce_eager": True,
            "gpu_memory_utilization": 0.4,
            "max_model_len": 512,
            "max_num_seqs": GROUP_SIZE,
            "disable_log_stats": True,
        },
    )
    engine = RolloutEngine(
        backend=backend,
        workflow=SingleTurnWorkflow(),
        reward_function=reward,
        config=RolloutConfig(
            group_size=GROUP_SIZE,
            train_batch_size=1,
            max_concurrent_groups=1,
        ),
    )
    try:
        await engine.rollout(prompt_source())
        for step in range(1, len(prompts) + 1):
            batch = await engine.next_batch()

            optimizer.zero_grad(set_to_none=True)
            loss = clipped_grpo_loss(trainer, batch)
            loss.backward()
            optimizer.step()

            checkpoint = run_directory / f"step-{step:04d}"
            trainer.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            policy_version = await engine.update_weights(
                VllmCheckpointWeightUpdate(checkpoint)
            )
            print(
                f"step={step} loss={loss.item():.4f} "
                f"policy_version={policy_version} checkpoint={checkpoint}"
            )

            if step < len(prompts):
                release_next_prompt[step - 1].set()
    finally:
        await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
