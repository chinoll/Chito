"""GSM8K GRPO with Accelerate, DeepSpeed, and one vLLM GPU.

For eight visible GPUs, start seven training ranks and leave the last GPU to
vLLM:

    accelerate launch --use_deepspeed --num_processes=7 --gpu_ids=all \
        samples/gsm8k_grpo.py
"""

from __future__ import annotations

import asyncio
import os
import re
from decimal import Decimal, InvalidOperation

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType, broadcast_object_list
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chito import (
    RolloutConfig,
    RolloutEngine,
    RolloutPrompt,
    RolloutSample,
    SingleTurnWorkflow,
    TrainingBatch,
    VllmBackend,
)


MODEL = os.environ.get("CHITO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GSM8K = os.environ.get("CHITO_GSM8K", "openai/gsm8k")
OUTPUT_DIR = os.environ.get("CHITO_OUTPUT_DIR", "outputs/gsm8k-grpo")
STEPS = int(os.environ.get("CHITO_STEPS", "500"))
PROMPTS_PER_STEP = int(os.environ.get("CHITO_PROMPTS_PER_STEP", "64"))
GROUP_SIZE = int(os.environ.get("CHITO_GROUP_SIZE", "4"))
MAX_NEW_TOKENS = int(os.environ.get("CHITO_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("CHITO_TEMPERATURE", "0.7"))
LEARNING_RATE = float(os.environ.get("CHITO_LEARNING_RATE", "1e-6"))
CLIP_EPSILON = 0.2
FORMAT_REWARD = 0.5
ANSWER_REWARD = 1.0

ANSWER_NUMBER = r"[-+]?\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\.?"
FORMAT_PATTERN = re.compile(
    rf"<think>.+?</think>\n<answer>\s*{ANSWER_NUMBER}\s*</answer>", re.DOTALL
)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER_PATTERN = re.compile(ANSWER_NUMBER)


def parse_number(text: str) -> Decimal | None:
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.removesuffix(".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def extract_number(text: str) -> Decimal | None:
    numbers = NUMBER_PATTERN.findall(text)
    return parse_number(numbers[-1]) if numbers else None


def score_output(output: str, target: str) -> tuple[float, float]:
    format_reward = FORMAT_REWARD * bool(FORMAT_PATTERN.fullmatch(output))
    answer_matches = ANSWER_PATTERN.findall(output)
    answer_text = answer_matches[-1] if answer_matches else output
    predicted = extract_number(answer_text)
    answer_reward = ANSWER_REWARD * (
        predicted is not None and predicted == extract_number(target)
    )
    return format_reward, answer_reward


def clipped_grpo_loss(
    model: torch.nn.Module,
    batch: TrainingBatch,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    samples: list[RolloutSample] = []
    advantages: list[torch.Tensor] = []
    for group in batch.groups:
        rewards = torch.tensor(
            [item.reward for item in group.samples],
            dtype=torch.float32,
            device=device,
        )
        group_advantages = (rewards - rewards.mean()) / (
            rewards.std(unbiased=False) + 1e-4
        )
        samples.extend(group.samples)
        advantages.extend(group_advantages.unbind())

    local_losses: list[torch.Tensor] = []
    for sample_index in range(rank, len(samples), world_size):
        sample = samples[sample_index]
        token_ids = torch.tensor(sample.token_ids, device=device)
        logits = model(token_ids[:-1].unsqueeze(0)).logits[0]
        next_tokens = token_ids[1:]
        current_logprobs = torch.log_softmax(logits.float(), dim=-1)
        current_logprobs = current_logprobs.gather(
            -1, next_tokens.unsqueeze(-1)
        ).squeeze(-1)

        generated_mask = torch.tensor(
            sample.loss_mask[1:], dtype=torch.bool, device=device
        )
        current_logprobs = current_logprobs[generated_mask]

        # There is one optimizer update per rollout batch, so the old policy is
        # this policy before backward, not vLLM's numerically different logits.
        old_logprobs = current_logprobs.detach()
        ratio = torch.exp(current_logprobs - old_logprobs)
        advantage = advantages[sample_index]
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantage
        local_losses.append(-torch.minimum(unclipped, clipped).mean())

    return torch.stack(local_losses).sum() * world_size / len(samples)


async def main() -> None:
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
    accelerator = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed_plugin,
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if accelerator.distributed_type is not DistributedType.DEEPSPEED:
        raise RuntimeError("launch this sample with accelerate --use_deepspeed")
    if world_size < 2:
        raise RuntimeError("GRPO needs at least two training ranks")
    if GROUP_SIZE * PROMPTS_PER_STEP < world_size:
        raise RuntimeError("rollout batch must cover every training rank")
    if torch.cuda.device_count() != world_size + 1:
        raise RuntimeError(
            "expose exactly one more GPU than Accelerate training processes"
        )

    device = accelerator.device
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    rollout_engine: RolloutEngine | None = None
    release_next_batch: list[asyncio.Event] = []
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        dataset = load_dataset(GSM8K, "main", split="train")
        dataset = dataset.shuffle(seed=42)

        prompts: list[RolloutPrompt] = []
        for index in range(STEPS * PROMPTS_PER_STEP):
            example = dataset[index % len(dataset)]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Solve the math problem. Return only this format:\n"
                        "<think>reasoning</think>\n"
                        "<answer>number</answer>\n"
                        "Put only the final number inside <answer> and write "
                        "nothing outside the tags."
                    ),
                },
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>2 plus 2 equals 4.</think>\n<answer>4</answer>"
                    ),
                },
                {"role": "user", "content": example["question"]},
            ]
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            prompts.append(
                RolloutPrompt(
                    prompt_id=f"gsm8k-{index}",
                    token_ids=tuple(token_ids),
                    metadata={
                        "target": example["answer"].rsplit("####", 1)[-1].strip()
                    },
                )
            )

        release_next_batch = [asyncio.Event() for _ in range(STEPS - 1)]

        async def prompt_source():
            for step in range(STEPS):
                if step > 0:
                    await release_next_batch[step - 1].wait()
                start = step * PROMPTS_PER_STEP
                for prompt in prompts[start : start + PROMPTS_PER_STEP]:
                    yield prompt

        async def reward(prompt: RolloutPrompt, sample: RolloutSample) -> float:
            output_ids = sample.token_ids[len(prompt.token_ids) :]
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            format_reward, answer_reward = score_output(
                output, str(prompt.metadata["target"])
            )
            return format_reward + answer_reward

        backend = VllmBackend(
            MODEL,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=1.0,
            weight_transfer="nccl",
            engine_kwargs={
                "dtype": "bfloat16",
                "device_ids": [world_size],
                "enforce_eager": True,
                "gpu_memory_utilization": 0.7,
                "max_model_len": 8192,
                "max_num_seqs": GROUP_SIZE * PROMPTS_PER_STEP,
                "disable_log_stats": True,
            },
        )
        rollout_engine = RolloutEngine(
            backend=backend,
            workflow=SingleTurnWorkflow(),
            reward_function=reward,
            config=RolloutConfig(
                group_size=GROUP_SIZE,
                train_batch_size=PROMPTS_PER_STEP,
                max_concurrent_groups=PROMPTS_PER_STEP,
            ),
        )
        await rollout_engine.rollout(prompt_source())

    try:
        for step in range(1, STEPS + 1):
            batch = await rollout_engine.next_batch() if rank == 0 else None
            objects = broadcast_object_list([batch], from_process=0)
            batch = objects[0]

            optimizer.zero_grad(set_to_none=True)
            loss = clipped_grpo_loss(model, batch, rank, world_size, device)
            accelerator.backward(loss)
            optimizer.step()

            mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
            torch.cuda.synchronize(device)
            accelerator.wait_for_everyone()

            if rank == 0:
                policy_version = await rollout_engine.update_weights(
                    accelerator.unwrap_model(model).named_parameters()
                )
                format_reward = 0.0
                answer_reward = 0.0
                zero_std_groups = 0
                sample_count = GROUP_SIZE * PROMPTS_PER_STEP
                for group in batch.groups:
                    target = str(group.prompt.metadata["target"])
                    rewards = [sample.reward for sample in group.samples]
                    zero_std_groups += len(set(rewards)) == 1
                    for sample in group.samples:
                        output_ids = sample.token_ids[len(group.prompt.token_ids) :]
                        output = tokenizer.decode(
                            output_ids, skip_special_tokens=True
                        ).strip()
                        sample_format, sample_answer = score_output(output, target)
                        format_reward += sample_format / sample_count
                        answer_reward += sample_answer / sample_count

                accelerator.print(
                    f"step={step} loss={mean_loss.item():.4f} "
                    f"format_rate={format_reward / FORMAT_REWARD:.3f} "
                    f"answer_accuracy={answer_reward / ANSWER_REWARD:.3f} "
                    f"total_reward={format_reward + answer_reward:.3f} "
                    f"zero_std={zero_std_groups / PROMPTS_PER_STEP:.3f} "
                    f"policy_version={policy_version}",
                    flush=True,
                )

            accelerator.wait_for_everyone()
            if rank == 0 and step < STEPS:
                release_next_batch[step - 1].set()

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if rank == 0:
            accelerator.unwrap_model(model).save_pretrained(
                OUTPUT_DIR,
                state_dict=state_dict,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(OUTPUT_DIR)
            accelerator.print(f"saved={OUTPUT_DIR}", flush=True)
    finally:
        if rollout_engine is not None:
            await rollout_engine.aclose()
        accelerator.end_training()


if __name__ == "__main__":
    asyncio.run(main())
