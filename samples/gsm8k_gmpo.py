"""Train Qwen2.5-0.5B on GSM8K with GMPO.

GMPO averages clipped token log-ratios before exponentiating them into one
geometric-mean ratio per completion. The old policy stays fixed while each
rollout batch receives two updates.

Example with four training GPUs and one rollout GPU:

    CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --use_deepspeed \
        --num_processes 4 samples/gsm8k_gmpo.py

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
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from chito import GRPOWorkflow, RolloutConfig, RolloutEngine, RolloutPrompt
from chito.gmpo import gmpo_loss


MODEL = os.environ.get("CHITO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GSM8K = os.environ.get("CHITO_GSM8K", "openai/gsm8k")
GSM8K_TRAIN = os.environ.get("CHITO_GSM8K_TRAIN")
OUTPUT_DIR = os.environ.get("CHITO_OUTPUT_DIR", "outputs/gsm8k-gmpo")
ROLLOUT_GPU_IDS = tuple(
    int(value) for value in os.environ.get("CHITO_ROLLOUT_GPU_IDS", "0").split(",")
)
STEPS = int(os.environ.get("CHITO_STEPS", "500"))
GROUP_SIZE = int(os.environ.get("CHITO_GROUP_SIZE", "4"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("CHITO_PER_DEVICE_BATCH_SIZE", "16"))
MAX_NEW_TOKENS = int(os.environ.get("CHITO_MAX_NEW_TOKENS", "1024"))
UPDATE_EPOCHS = int(os.environ.get("CHITO_UPDATE_EPOCHS", "2"))
TEMPERATURE = float(os.environ.get("CHITO_TEMPERATURE", "1.0"))
LEARNING_RATE = float(os.environ.get("CHITO_LEARNING_RATE", "1e-6"))
LOG_RATIO_CLIP = 0.4
FORMAT_REWARD = 0.5
ANSWER_REWARD = 1.0

ANSWER_NUMBER = r"[-+]?\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\.?"
FORMAT_PATTERN = re.compile(
    rf"<think>.+?</think><answer>\s*{ANSWER_NUMBER}\s*</answer>", re.DOTALL
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
    if not FORMAT_PATTERN.fullmatch(output):
        return 0.0, 0.0

    answer_text = ANSWER_PATTERN.findall(output)[-1]
    predicted_answer = extract_number(answer_text)
    target_answer = extract_number(target)
    answer_reward = (
        ANSWER_REWARD
        if predicted_answer is not None and predicted_answer == target_answer
        else 0.0
    )
    return FORMAT_REWARD, answer_reward


def decode_completion(tokenizer, sample) -> str:
    completion_ids = [
        token_id
        for token_id, selected in zip(sample.token_ids, sample.loss_mask, strict=True)
        if selected
    ]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


class GSM8KGMPOWorkflow(GRPOWorkflow):
    """GSM8K prompts and strictly formatted verifiable rewards."""

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
                    "<think>reasoning</think><answer>number</answer>\n"
                    "Put only the final number inside <answer> and write "
                    "nothing outside the tags."
                ),
            },
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>2 plus 2 equals 4.</think><answer>4</answer>",
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
        output = decode_completion(self.tokenizer, sample)
        return sum(score_output(output, str(prompt.metadata["target"])))


def right_pad(rows: list[torch.Tensor], value: float) -> torch.Tensor:
    return pad_sequence(rows, batch_first=True, padding_value=value)


def prepare_gmpo_tensors(batch, pad_token_id: int, device: torch.device):
    input_ids = right_pad(
        [torch.tensor(sample.token_ids, device=device) for sample in batch.samples],
        pad_token_id,
    )
    completion_mask = right_pad(
        [
            torch.tensor(sample.loss_mask[1:], dtype=torch.float32, device=device)
            for sample in batch.samples
        ],
        0,
    )
    advantages = torch.tensor(
        [sample.advantage for sample in batch.samples], device=device
    )
    return input_ids, completion_mask, advantages


def per_token_logprobs(model, input_ids):
    logits = model(input_ids, use_cache=False).logits[:, :-1]
    selected = []
    for row_logits, row_ids in zip(logits, input_ids[:, 1:], strict=True):
        row_logprobs = F.log_softmax(row_logits.float() / TEMPERATURE, dim=-1)
        selected.append(row_logprobs.gather(-1, row_ids.unsqueeze(-1)).squeeze(-1))
    return torch.stack(selected)


def batch_metrics(batch, tokenizer, device: torch.device) -> torch.Tensor:
    format_total = 0.0
    answer_total = 0.0
    reward_total = 0.0
    for sample in batch.samples:
        output = decode_completion(tokenizer, sample)
        format_reward, answer_reward = score_output(
            output, str(sample.metadata["target"])
        )
        format_total += format_reward
        answer_total += answer_reward
        reward_total += sample.reward
    return torch.tensor([format_total, answer_total, reward_total], device=device)


async def main() -> None:
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
        PER_DEVICE_BATCH_SIZE
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed_plugin,
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
    policy_model = accelerator.unwrap_model(model)

    prompts_per_step = PER_DEVICE_BATCH_SIZE * accelerator.num_processes // GROUP_SIZE
    dataset = None
    if accelerator.is_main_process:
        if GSM8K_TRAIN:
            dataset = load_dataset("json", data_files=GSM8K_TRAIN, split="train")
        else:
            dataset = load_dataset(GSM8K, "main", split="train")
    workflow = GSM8KGMPOWorkflow(MODEL) if accelerator.is_main_process else None
    rollout_config = RolloutConfig(
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
    engine = RolloutEngine(dataset=dataset, workflow=workflow, config=rollout_config)

    try:
        for step in range(1, STEPS + 1):
            batch = await engine.next_batch()
            input_ids, completion_mask, advantages = prepare_gmpo_tensors(
                batch,
                tokenizer.pad_token_id,
                accelerator.device,
            )
            with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                old_logprobs = per_token_logprobs(policy_model, input_ids).detach()

            for _ in range(UPDATE_EPOCHS):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    current_logprobs = per_token_logprobs(policy_model, input_ids)
                    loss, clip_metrics = gmpo_loss(
                        current_logprobs,
                        old_logprobs,
                        advantages,
                        completion_mask,
                        epsilon_low=LOG_RATIO_CLIP,
                        epsilon_high=LOG_RATIO_CLIP,
                    )
                    accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            reward_totals = batch_metrics(batch, tokenizer, accelerator.device)
            mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
            reward_totals = accelerator.reduce(reward_totals, reduction="sum")
            token_count = completion_mask.sum()
            clip_totals = accelerator.reduce(
                torch.stack(clip_metrics) * token_count,
                reduction="sum",
            )
            token_count = accelerator.reduce(token_count, reduction="sum")
            clip_metrics = clip_totals / token_count

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                policy_version = await engine.update_weights(policy_model)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                sample_count = batch.global_sample_count
                format_total, answer_total, reward_total = reward_totals.tolist()
                low_clip, high_clip, clip_ratio = clip_metrics.tolist()
                accelerator.print(
                    f"step={step} loss={mean_loss.item():.4f} "
                    f"format_rate={format_total / FORMAT_REWARD / sample_count:.3f} "
                    f"answer_accuracy={answer_total / ANSWER_REWARD / sample_count:.3f} "
                    f"total_reward={reward_total / sample_count:.3f} "
                    f"low_clip={low_clip:.4f} high_clip={high_clip:.4f} "
                    f"clip_ratio={clip_ratio:.4f} "
                    f"policy_version={policy_version}",
                    flush=True,
                )

        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            policy_model.save_pretrained(
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
