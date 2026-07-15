"""Train Qwen2.5-0.5B on GSM8K with GRPO, DeepSpeed, and vLLM.

Example with four training GPUs and one rollout GPU:

    CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --use_deepspeed \
        --num_processes 4 samples/gsm8k_grpo.py

Set ``CHITO_ROLLOUT_GPU_IDS`` to the physical GPU used by vLLM. It does not
need to be visible to the Accelerate training processes.
"""

from __future__ import annotations

import asyncio
import os
import re
from decimal import Decimal, InvalidOperation

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chito import GRPOWorkflow, RolloutConfig, RolloutEngine, RolloutPrompt


MODEL = os.environ.get("CHITO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GSM8K = os.environ.get("CHITO_GSM8K", "openai/gsm8k")
GSM8K_TRAIN = os.environ.get("CHITO_GSM8K_TRAIN")
OUTPUT_DIR = os.environ.get("CHITO_OUTPUT_DIR", "outputs/gsm8k-grpo")
ROLLOUT_GPU_IDS = tuple(
    int(value) for value in os.environ.get("CHITO_ROLLOUT_GPU_IDS", "0").split(",")
)
STEPS = int(os.environ.get("CHITO_STEPS", "500"))
GROUP_SIZE = int(os.environ.get("CHITO_GROUP_SIZE", "4"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("CHITO_PER_DEVICE_BATCH_SIZE", "64"))
MAX_NEW_TOKENS = int(os.environ.get("CHITO_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("CHITO_TEMPERATURE", "0.7"))
LEARNING_RATE = float(os.environ.get("CHITO_LEARNING_RATE", "1e-6"))
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


class GSM8KWorkflow(GRPOWorkflow):
    """GSM8K prompt construction and verifiable rewards."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.tokenizer = None

    async def setup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    async def prepare(self, item: object, item_id: str) -> RolloutPrompt:
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
                "content": "<think>2 plus 2 equals 4.</think>\n<answer>4</answer>",
            },
            {"role": "user", "content": item["question"]},
        ]
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        target = item["answer"].rsplit("####", 1)[-1].strip()
        return RolloutPrompt(item_id, tuple(token_ids), {"target": target})

    async def reward(self, prompt, sample) -> float:
        output_ids = [
            token_id
            for token_id, selected in zip(
                sample.token_ids, sample.loss_mask, strict=True
            )
            if selected
        ]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return sum(score_output(output, str(prompt.metadata["target"])))


def left_pad(rows: list[torch.Tensor], value: float) -> torch.Tensor:
    width = max(row.size(0) for row in rows)
    return torch.stack(
        [F.pad(row, (width - row.size(0), 0), value=value) for row in rows]
    )


def right_pad(rows: list[torch.Tensor], value: float) -> torch.Tensor:
    width = max(row.size(0) for row in rows)
    return torch.stack(
        [F.pad(row, (0, width - row.size(0)), value=value) for row in rows]
    )


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """TRL's memory-conscious log_softmax followed by gather."""
    selected = []
    for row_logits, row_labels in zip(logits, labels, strict=True):
        row_logprobs = F.log_softmax(row_logits, dim=-1)
        selected.append(row_logprobs.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(selected)


def grpo_loss(model, batch, pad_token_id: int, device: torch.device):
    completion_starts = [sample.loss_mask.index(True) for sample in batch.samples]
    prompt_ids = left_pad(
        [
            torch.tensor(sample.token_ids[:start], device=device)
            for sample, start in zip(batch.samples, completion_starts, strict=True)
        ],
        pad_token_id,
    )
    completion_ids = right_pad(
        [
            torch.tensor(sample.token_ids[start:], device=device)
            for sample, start in zip(batch.samples, completion_starts, strict=True)
        ],
        pad_token_id,
    )
    prompt_mask = left_pad(
        [torch.ones(start, device=device) for start in completion_starts], 0
    )
    completion_attention_mask = right_pad(
        [
            torch.ones(len(sample.token_ids) - start, device=device)
            for sample, start in zip(batch.samples, completion_starts, strict=True)
        ],
        0,
    )
    completion_mask = right_pad(
        [
            torch.tensor(
                sample.loss_mask[start:], dtype=torch.float32, device=device
            )
            for sample, start in zip(batch.samples, completion_starts, strict=True)
        ],
        0,
    )
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_attention_mask], dim=1)
    advantages = torch.tensor(
        [sample.advantage for sample in batch.samples], device=device
    ).unsqueeze(1)

    completion_width = completion_ids.size(1)
    logits_to_keep = completion_width + 1
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep,
        use_cache=False,
    ).logits[:, :-1]
    logits.div_(TEMPERATURE)
    labels = input_ids[:, -completion_width:]
    mask = completion_mask
    per_token_logprobs = selective_log_softmax(logits, labels)

    # TRL v0.15.2 with beta=0 and num_iterations=1 uses this forward pass
    # detached as the old policy.
    ratio = torch.exp(per_token_logprobs - per_token_logprobs.detach())
    per_token_loss = -ratio * advantages

    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()
    return loss


def batch_metrics(batch, tokenizer, device: torch.device) -> torch.Tensor:
    format_total = 0.0
    answer_total = 0.0
    reward_total = 0.0
    for sample in batch.samples:
        output_ids = [
            token_id
            for token_id, selected in zip(
                sample.token_ids, sample.loss_mask, strict=True
            )
            if selected
        ]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        format_reward, answer_reward = score_output(
            output, str(sample.metadata["target"])
        )
        format_total += format_reward
        answer_total += answer_reward
        reward_total += sample.reward
    return torch.tensor([format_total, answer_total, reward_total], device=device)


async def main() -> None:
    deepspeed = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
    )
    deepspeed.deepspeed_config["train_micro_batch_size_per_gpu"] = PER_DEVICE_BATCH_SIZE
    accelerator = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    prompts_per_step = PER_DEVICE_BATCH_SIZE * accelerator.num_processes // GROUP_SIZE
    dataset = None
    if accelerator.is_main_process:
        dataset = (
            load_dataset("json", data_files=GSM8K_TRAIN, split="train")
            if GSM8K_TRAIN
            else load_dataset(GSM8K, "main", split="train")
        )
    workflow = GSM8KWorkflow(MODEL) if accelerator.is_main_process else None
    config = RolloutConfig(
        model=MODEL,
        group_size=GROUP_SIZE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        max_concurrent_groups=prompts_per_step,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        rollout_gpu_ids=ROLLOUT_GPU_IDS,
        backend_kwargs={
            "dtype": "bfloat16",
            "enforce_eager": True,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 4096,
            "max_num_seqs": prompts_per_step * GROUP_SIZE,
            "disable_log_stats": True,
        },
    )
    engine = RolloutEngine(dataset=dataset, workflow=workflow, config=config)

    try:
        for step in range(1, STEPS + 1):
            batch = await engine.next_batch()
            loss = grpo_loss(
                model, batch, tokenizer.pad_token_id, accelerator.device
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            metrics = batch_metrics(batch, tokenizer, accelerator.device)
            loss = accelerator.reduce(loss.detach(), reduction="mean")
            metrics = accelerator.reduce(metrics, reduction="sum")

            accelerator.wait_for_everyone()
            policy_version = None
            if accelerator.is_main_process:
                policy_version = await engine.update_weights(
                    accelerator.unwrap_model(model)
                )
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                sample_count = batch.global_sample_count
                format_rate = metrics[0].item() / FORMAT_REWARD / sample_count
                answer_accuracy = metrics[1].item() / ANSWER_REWARD / sample_count
                total_reward = metrics[2].item() / sample_count
                accelerator.print(
                    f"step={step} loss={loss.item():.4f} "
                    f"format_rate={format_rate:.3f} "
                    f"answer_accuracy={answer_accuracy:.3f} "
                    f"total_reward={total_reward:.3f} "
                    f"policy_version={policy_version}",
                    flush=True,
                )

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                OUTPUT_DIR,
                state_dict=state_dict,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(OUTPUT_DIR)
            accelerator.print(f"saved={OUTPUT_DIR}", flush=True)
    finally:
        await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
