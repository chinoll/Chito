"""Train Qwen2.5-0.5B on GSM8K with synchronous or asynchronous GRPO.
    CHITO_ROLLOUT_MODE=async CUDA_VISIBLE_DEVICES=2,3,4,5 \
        accelerate launch --use_deepspeed --num_processes 4 \
        samples/async_gsm8k_grpo.py
Set ``CHITO_ROLLOUT_GPU_IDS`` to the physical GPU used by vLLM.
"""

from __future__ import annotations

import asyncio
import os
import re
from decimal import Decimal, InvalidOperation

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from chito import GRPOWorkflow, RolloutConfig, RolloutEngine, RolloutPrompt


MODEL = os.environ.get("CHITO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GSM8K = os.environ.get("CHITO_GSM8K", "openai/gsm8k")
GSM8K_TRAIN = os.environ.get("CHITO_GSM8K_TRAIN")
OUTPUT_DIR = os.environ.get("CHITO_OUTPUT_DIR", "outputs/async-gsm8k-grpo")
ROLLOUT_MODE = os.environ.get("CHITO_ROLLOUT_MODE", "async")
ROLLOUT_GPU_IDS = tuple(
    int(value) for value in os.environ.get("CHITO_ROLLOUT_GPU_IDS", "0").split(",")
)
STEPS = int(os.environ.get("CHITO_STEPS", "500"))
GROUP_SIZE = int(os.environ.get("CHITO_GROUP_SIZE", "4"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("CHITO_PER_DEVICE_BATCH_SIZE", "32"))
MAX_NEW_TOKENS = int(os.environ.get("CHITO_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("CHITO_TEMPERATURE", "1.0"))
LEARNING_RATE = float(os.environ.get("CHITO_LEARNING_RATE", "1e-6"))
SEED = int(os.environ.get("CHITO_SEED", "42"))
CLIP_EPSILON = 0.2
TIS_CLIP_MAX = 3.0
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
    format_reward = FORMAT_REWARD if FORMAT_PATTERN.fullmatch(output) else 0.0

    answer_matches = ANSWER_PATTERN.findall(output)
    answer_text = answer_matches[-1] if answer_matches else output
    predicted_answer = extract_number(answer_text)
    target_answer = extract_number(target)
    answer_is_correct = (
        predicted_answer is not None and predicted_answer == target_answer
    )
    answer_reward = ANSWER_REWARD if answer_is_correct else 0.0
    return format_reward, answer_reward


def decode_completion(tokenizer, sample) -> str:
    completion_ids = [
        token_id
        for token_id, selected in zip(sample.token_ids, sample.loss_mask, strict=True)
        if selected
    ]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


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
        output = decode_completion(self.tokenizer, sample)
        return sum(score_output(output, str(prompt.metadata["target"])))


def right_pad(rows: list[torch.Tensor], value: float) -> torch.Tensor:
    return pad_sequence(rows, batch_first=True, padding_value=value)


def prepare_grpo_tensors(batch, pad_token_id: int, device: torch.device):
    input_ids = right_pad(
        [torch.tensor(sample.token_ids, device=device) for sample in batch.samples],
        pad_token_id,
    )
    loss_mask = right_pad(
        [
            torch.tensor(sample.loss_mask[1:], dtype=torch.float32, device=device)
            for sample in batch.samples
        ],
        0,
    )
    behavior_logprobs = right_pad(
        [
            torch.tensor(
                sample.behavior_logprobs[1:], dtype=torch.float32, device=device
            )
            for sample in batch.samples
        ],
        0,
    )
    advantages = torch.tensor(
        [sample.advantage for sample in batch.samples],
        dtype=torch.float32,
        device=device,
    )
    return input_ids, loss_mask, behavior_logprobs, advantages


def per_token_logprobs(model, input_ids, temperature):
    logits = (
        model(
            input_ids=input_ids,
            use_cache=False,
        )
        .logits[:, :-1]
        .float()
    )
    labels = input_ids[:, 1:]
    return (
        F.log_softmax(logits / temperature, dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )


def grpo_tis_loss(
    current_logprobs,
    behavior_logprobs,
    loss_mask,
    advantages,
):
    selected = loss_mask.bool()
    proximal_old_logprobs = current_logprobs.detach()
    proximal_log_ratio = current_logprobs.float() - proximal_old_logprobs.float()
    proximal_ratio = torch.exp(proximal_log_ratio)
    correction_log_ratio = proximal_old_logprobs.float() - behavior_logprobs.float()
    raw_correction_ratio = torch.exp(correction_log_ratio)
    tis_weight = raw_correction_ratio.clamp(max=TIS_CLIP_MAX).detach()

    unclipped = proximal_ratio * advantages[:, None]
    clipped = (
        proximal_ratio.clamp(1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages[:, None]
    )
    per_token_loss = -tis_weight * torch.minimum(unclipped, clipped)
    loss = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1)).mean()
    has_nonfinite = (
        ~torch.isfinite(current_logprobs[selected]).all()
        | ~torch.isfinite(behavior_logprobs[selected]).all()
        | ~torch.isfinite(correction_log_ratio[selected]).all()
        | ~torch.isfinite(raw_correction_ratio[selected]).all()
        | ~torch.isfinite(per_token_loss[selected]).all()
        | ~torch.isfinite(loss)
    )
    return loss, has_nonfinite.to(torch.int64)


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
    set_seed(SEED)

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
    workflow = GSM8KWorkflow(MODEL) if accelerator.is_main_process else None
    rollout_config = RolloutConfig(
        model=MODEL,
        group_size=GROUP_SIZE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        max_concurrent_groups=prompts_per_step,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        rollout_mode=ROLLOUT_MODE,
        rollout_gpu_ids=ROLLOUT_GPU_IDS,
        seed=SEED,
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

    learner_policy_version = rollout_config.initial_policy_version
    engine_closed = False

    try:
        for step in range(1, STEPS + 1):
            batch = await engine.next_batch()
            proximal_policy_version = learner_policy_version
            input_ids, loss_mask, behavior_logprobs, advantages = prepare_grpo_tensors(
                batch,
                tokenizer.pad_token_id,
                accelerator.device,
            )

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                current_logprobs = per_token_logprobs(
                    model,
                    input_ids,
                    TEMPERATURE,
                )
                loss, nonfinite_count = grpo_tis_loss(
                    current_logprobs,
                    behavior_logprobs,
                    loss_mask,
                    advantages,
                )
                if accelerator.num_processes > 1:
                    torch.distributed.all_reduce(nonfinite_count)
                if nonfinite_count.item():
                    raise FloatingPointError(
                        "GRPO loss input or output contains NaN or Inf"
                    )
                accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            reward_totals = accelerator.reduce(
                batch_metrics(batch, tokenizer, accelerator.device),
                reduction="sum",
            )
            mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
            policy_lag = proximal_policy_version - batch.policy_version

            if step < STEPS:
                accelerator.wait_for_everyone()
                updated_version = 0
                update_error = None
                if accelerator.is_main_process:
                    try:
                        updated_version = await engine.update_weights(policy_model)
                    except BaseException as exc:
                        update_error = exc

                update_status = torch.tensor(
                    [update_error is None, updated_version],
                    dtype=torch.int64,
                    device=accelerator.device,
                )
                if accelerator.num_processes > 1:
                    torch.distributed.broadcast(update_status, src=0)
                if not update_status[0].item():
                    message = [
                        f"{type(update_error).__name__}: {update_error}"
                        if update_error is not None
                        else None
                    ]
                    if accelerator.num_processes > 1:
                        torch.distributed.broadcast_object_list(
                            message,
                            src=0,
                            device=accelerator.device,
                        )
                    if update_error is not None:
                        raise update_error
                    raise RuntimeError(f"rank 0 weight update failed: {message[0]}")
                learner_policy_version = int(update_status[1].item())

            if accelerator.is_main_process:
                sample_count = batch.global_sample_count
                format_total, answer_total, reward_total = reward_totals.tolist()
                accelerator.print(
                    f"step={step} mode={ROLLOUT_MODE} "
                    f"loss={mean_loss.item():.6f} "
                    f"format_rate={format_total / FORMAT_REWARD / sample_count:.6f} "
                    f"answer_accuracy={answer_total / ANSWER_REWARD / sample_count:.6f} "
                    f"reward={reward_total / sample_count:.6f} "
                    f"policy_lag={policy_lag} "
                    f"behavior_policy_version={batch.policy_version} "
                    f"proximal_policy_version={proximal_policy_version} "
                    f"committed_policy_version={learner_policy_version}",
                    flush=True,
                )

            if step == STEPS:
                accelerator.wait_for_everyone()
                close_error = None
                try:
                    await engine.aclose()
                except BaseException as exc:
                    close_error = exc
                engine_closed = True
                close_status = torch.tensor(
                    [close_error is None],
                    dtype=torch.int64,
                    device=accelerator.device,
                )
                if accelerator.num_processes > 1:
                    torch.distributed.all_reduce(
                        close_status,
                        op=torch.distributed.ReduceOp.MIN,
                    )
                if not close_status.item():
                    if close_error is not None:
                        raise close_error
                    raise RuntimeError("rollout engine close failed on another rank")
    finally:
        if not engine_closed:
            await engine.aclose()

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        policy_model.save_pretrained(
            OUTPUT_DIR,
            state_dict=state_dict,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(OUTPUT_DIR)
        accelerator.print(f"saved={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
