from __future__ import annotations

import asyncio
import math
import os

import pytest

from chito import (
    RolloutConfig,
    RolloutEngine,
    RolloutPrompt,
    RolloutSample,
    SingleTurnWorkflow,
    VllmBackend,
)


RUN_INTEGRATION = os.environ.get("CHITO_RUN_VLLM_INTEGRATION") == "1"
MODEL = os.environ.get(
    "CHITO_VLLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="set CHITO_RUN_VLLM_INTEGRATION=1 to load a real vLLM model",
    ),
]


async def one_prompt(prompt: RolloutPrompt):
    yield prompt


async def sample_index_reward(
    prompt: RolloutPrompt, sample: RolloutSample
) -> float:
    return float(sample.sample_index)


def test_qwen_half_billion_model_produces_grpo_batch() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one short greeting."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert isinstance(prompt_ids, list)

    async def scenario() -> None:
        backend = VllmBackend(
            MODEL,
            max_tokens=int(os.environ.get("CHITO_VLLM_MAX_TOKENS", "8")),
            temperature=0.8,
            engine_kwargs={
                "dtype": os.environ.get("CHITO_VLLM_DTYPE", "float16"),
                "enforce_eager": True,
                "gpu_memory_utilization": float(
                    os.environ.get("CHITO_VLLM_GPU_MEMORY_UTILIZATION", "0.5")
                ),
                "max_model_len": int(
                    os.environ.get("CHITO_VLLM_MAX_MODEL_LEN", "512")
                ),
                "max_num_seqs": 2,
                "disable_log_stats": True,
            },
        )
        engine = RolloutEngine(
            backend=backend,
            workflow=SingleTurnWorkflow(),
            reward_function=sample_index_reward,
            config=RolloutConfig(
                group_size=2,
                train_batch_size=1,
                max_concurrent_groups=1,
            ),
        )
        prompt = RolloutPrompt("qwen-smoke", tuple(prompt_ids))

        try:
            await engine.rollout(one_prompt(prompt))
            batch = await engine.next_batch()
        finally:
            await engine.aclose()

        assert len(batch.groups) == 1
        group = batch.groups[0]
        assert len(group.samples) == 2
        assert {sample.reward for sample in group.samples} == {0.0, 1.0}
        for sample in group.samples:
            assert sample.token_ids[: len(prompt_ids)] == tuple(prompt_ids)
            generated_ids = sample.token_ids[len(prompt_ids) :]
            generated_logprobs = sample.logprobs[len(prompt_ids) :]
            assert generated_ids
            assert len(generated_ids) == len(generated_logprobs)
            assert all(math.isfinite(value) for value in generated_logprobs)
            assert sample.loss_mask == (False,) * len(prompt_ids) + (
                True,
            ) * len(generated_ids)

    asyncio.run(scenario())
