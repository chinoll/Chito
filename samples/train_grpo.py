"""A minimal one-training-GPU GRPO loop with a second GPU for vLLM."""

from __future__ import annotations

import asyncio
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from chito import GRPOWorkflow, RolloutConfig, RolloutEngine, RolloutPrompt


MODEL = os.environ.get("CHITO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ROLLOUT_GPU = int(os.environ.get("CHITO_ROLLOUT_GPU", "1"))
DEVICE = torch.device("cuda:0")
GROUP_SIZE = 4
MAX_NEW_TOKENS = 24
LEARNING_RATE = 1e-5
TASKS = (
    {"question": "Name the color of a clear daytime sky.", "target": "blue"},
    {"question": "Write two plus two in words.", "target": "four"},
    {"question": "Name the opposite of cold.", "target": "warm"},
)


class ToyWorkflow(GRPOWorkflow):
    def __init__(self, model: str) -> None:
        self.model = model
        self.tokenizer = None

    async def setup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    async def prepare(self, item: object, item_id: str) -> RolloutPrompt:
        token_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        return RolloutPrompt(item_id, tuple(token_ids), {"target": item["target"]})

    async def reward(self, prompt, sample) -> float:
        output_ids = [
            token_id
            for token_id, selected in zip(
                sample.token_ids, sample.loss_mask, strict=True
            )
            if selected
        ]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).lower()
        return float(str(prompt.metadata["target"]).lower() in output)


def left_pad(rows: list[torch.Tensor], value: float) -> torch.Tensor:
    width = max(row.size(0) for row in rows)
    return torch.stack(
        [F.pad(row, (width - row.size(0), 0), value=value) for row in rows]
    )


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    selected = []
    for row_logits, row_labels in zip(logits, labels, strict=True):
        row_logprobs = F.log_softmax(row_logits, dim=-1)
        selected.append(row_logprobs.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(selected)


def grpo_loss(model, batch, pad_token_id: int) -> torch.Tensor:
    input_ids = left_pad(
        [torch.tensor(sample.token_ids, device=DEVICE) for sample in batch.samples],
        pad_token_id,
    )
    attention_mask = left_pad(
        [torch.ones(len(sample.token_ids), device=DEVICE) for sample in batch.samples],
        0,
    )
    completion_mask = left_pad(
        [
            torch.tensor(sample.loss_mask, dtype=torch.float32, device=DEVICE)
            for sample in batch.samples
        ],
        0,
    )
    advantages = torch.tensor(
        [sample.advantage for sample in batch.samples], device=DEVICE
    ).unsqueeze(1)

    completion_width = int(completion_mask.sum(-1).max().item())
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        logits_to_keep=completion_width + 1,
        use_cache=False,
    ).logits[:, :-1]
    labels = input_ids[:, -completion_width:]
    mask = completion_mask[:, -completion_width:]
    per_token_logprobs = selective_log_softmax(logits, labels)

    ratio = torch.exp(per_token_logprobs - per_token_logprobs.detach())
    per_token_loss = -ratio * advantages
    return ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()


async def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    engine = RolloutEngine(
        dataset=TASKS,
        workflow=ToyWorkflow(MODEL),
        config=RolloutConfig(
            model=MODEL,
            group_size=GROUP_SIZE,
            per_device_train_batch_size=GROUP_SIZE,
            max_concurrent_groups=1,
            max_tokens=MAX_NEW_TOKENS,
            rollout_gpu_ids=(ROLLOUT_GPU,),
            backend_kwargs={
                "dtype": "bfloat16",
                "enforce_eager": True,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 512,
                "disable_log_stats": True,
            },
        ),
    )

    try:
        for step in range(1, len(TASKS) + 1):
            batch = await engine.next_batch()
            optimizer.zero_grad(set_to_none=True)
            loss = grpo_loss(model, batch, tokenizer.pad_token_id)
            loss.backward()
            optimizer.step()

            policy_version = await engine.update_weights(model)
            reward = sum(sample.reward for sample in batch.samples) / len(batch.samples)
            print(
                f"step={step} loss={loss.item():.4f} "
                f"reward={reward:.3f} policy_version={policy_version}",
                flush=True,
            )
    finally:
        await engine.aclose()

    tokenizer.save_pretrained("outputs/toy-grpo")
    model.save_pretrained("outputs/toy-grpo")


if __name__ == "__main__":
    asyncio.run(main())
