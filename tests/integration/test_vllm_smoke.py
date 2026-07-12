from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping
from tempfile import TemporaryDirectory

import pytest

from chito import (
    RolloutConfig,
    RolloutEngine,
    RolloutPrompt,
    RolloutSample,
    SingleTurnWorkflow,
    VllmBackend,
    VllmWeightUpdate,
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


async def gated_prompts(
    prompt_ids: tuple[int, ...], release_second: asyncio.Event
):
    yield RolloutPrompt("qwen-before-update", prompt_ids)
    await release_second.wait()
    yield RolloutPrompt("qwen-after-update", prompt_ids)


async def sample_index_reward(
    prompt: RolloutPrompt, sample: RolloutSample
) -> float:
    return float(sample.sample_index)


def test_qwen_half_billion_model_rollout_and_weight_update() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one short greeting."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(rendered_prompt, Mapping):
        prompt_ids = list(rendered_prompt["input_ids"])
    else:
        prompt_ids = rendered_prompt
    assert isinstance(prompt_ids, list)

    async def scenario() -> None:
        backend = VllmBackend(
            MODEL,
            max_tokens=1,
            temperature=0.0,
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
        release_second = asyncio.Event()

        try:
            await engine.rollout(
                gated_prompts(tuple(prompt_ids), release_second)
            )
            before_batch = await engine.next_batch()

            trainer = AutoModelForCausalLM.from_pretrained(
                MODEL,
                dtype=torch.float16,
            ).to("cuda:0")
            trainer.requires_grad_(False)
            with torch.no_grad():
                for parameter in trainer.parameters():
                    parameter.zero_()

            with TemporaryDirectory(prefix="chito-zero-checkpoint-") as path:
                trainer.save_pretrained(path, safe_serialization=True)
                del trainer
                torch.cuda.empty_cache()

                update = VllmWeightUpdate(path)
                assert await engine.update_weights(update) == 1
            del update
            torch.cuda.empty_cache()

            release_second.set()
            after_batch = await engine.next_batch()
        finally:
            await engine.aclose()

        assert len(before_batch.groups) == len(after_batch.groups) == 1
        before_group = before_batch.groups[0]
        after_group = after_batch.groups[0]
        assert before_group.policy_version == 0
        assert after_group.policy_version == 1

        before_tokens = _assert_group(before_group.samples, prompt_ids)
        after_tokens = _assert_group(after_group.samples, prompt_ids)
        assert all(token_id != 0 for token_id in before_tokens)
        assert after_tokens == [0, 0]

    asyncio.run(scenario())


def _assert_group(
    samples: tuple[RolloutSample, ...], prompt_ids: list[int]
) -> list[int]:
    assert len(samples) == 2
    assert {sample.reward for sample in samples} == {0.0, 1.0}
    generated_tokens: list[int] = []
    for sample in samples:
        assert sample.token_ids[: len(prompt_ids)] == tuple(prompt_ids)
        generated_ids = sample.token_ids[len(prompt_ids) :]
        generated_logprobs = sample.logprobs[len(prompt_ids) :]
        assert len(generated_ids) == len(generated_logprobs) == 1
        assert math.isfinite(generated_logprobs[0])
        assert sample.loss_mask == (False,) * len(prompt_ids) + (True,)
        generated_tokens.append(generated_ids[0])
    return generated_tokens
