from __future__ import annotations

import asyncio
import pickle
from dataclasses import replace

import pytest

from chito import (
    EngineClosedError,
    InvalidRolloutGroupError,
    RolloutAlreadyStartedError,
    RolloutConfig,
    RolloutEngine,
    RolloutFailedError,
    RolloutGroup,
    RolloutPrompt,
    RolloutSample,
    SingleTurnWorkflow,
    SourceExhaustedError,
)

from .fakes import FakeInferenceBackend, wait_until


async def prompts(*values: RolloutPrompt):
    for value in values:
        yield value


async def metadata_reward(
    prompt: RolloutPrompt, sample: RolloutSample
) -> float:
    return float(prompt.metadata["target"]) + sample.sample_index


def test_rollout_preserves_exact_training_data() -> None:
    async def scenario() -> None:
        backend = FakeInferenceBackend()
        config = RolloutConfig(
            group_size=2,
            train_batch_size=2,
            max_concurrent_groups=2,
        )
        engine = RolloutEngine(
            backend, SingleTurnWorkflow(), metadata_reward, config
        )
        prompt_a = RolloutPrompt("a", (1, 2), {"target": 3})
        prompt_b = RolloutPrompt("b", (4,), {"target": 7})

        await engine.rollout(prompts(prompt_a, prompt_b))
        batch = await engine.next_batch()

        assert len(batch.groups) == 2
        groups = {group.prompt.prompt_id: group for group in batch.groups}
        assert set(groups) == {"a", "b"}
        for prompt in (prompt_a, prompt_b):
            group = groups[prompt.prompt_id]
            assert group.policy_version == 0
            assert len(group.samples) == 2
            for sample in group.samples:
                expected_output = (
                    1000 + sample.sample_index * 10,
                    1001 + sample.sample_index * 10,
                )
                assert sample.token_ids == prompt.token_ids + expected_output
                assert sample.logprobs == (0.0,) * len(prompt.token_ids) + (
                    -0.25,
                    -0.5,
                )
                assert sample.loss_mask == (False,) * len(prompt.token_ids) + (
                    True,
                    True,
                )
                assert sample.reward == float(prompt.metadata["target"]) + (
                    sample.sample_index
                )
                assert sample.policy_version == 0

        with pytest.raises(SourceExhaustedError) as exhausted:
            await engine.next_batch()
        assert exhausted.value.remaining_groups == ()
        await engine.aclose()

    asyncio.run(scenario())


def test_prompt_metadata_is_copied() -> None:
    metadata = {"target": 1}
    prompt = RolloutPrompt("p", (1,), metadata)
    metadata["target"] = 99
    assert prompt.metadata["target"] == 1

    restored = pickle.loads(pickle.dumps(prompt))
    assert restored == prompt
    assert isinstance(restored.metadata, dict)


def test_grpo_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        RolloutConfig(
            group_size=1,
            train_batch_size=1,
            max_concurrent_groups=1,
        )


def test_agent_workflow_may_mask_tool_context_tokens() -> None:
    class AgentWorkflow:
        async def run(self, context, prompt):
            return RolloutSample(
                prompt_id=prompt.prompt_id,
                sample_index=context.sample_index,
                token_ids=prompt.token_ids + (10, 20),
                logprobs=(0.0,) * len(prompt.token_ids) + (-0.1, 0.0),
                loss_mask=(False,) * len(prompt.token_ids) + (True, False),
                policy_version=context.policy_version,
            )

    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(),
            AgentWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )
        await engine.rollout(
            prompts(RolloutPrompt("agent", (1,), {"target": 1}))
        )
        batch = await engine.next_batch()
        assert batch.groups[0].samples[0].loss_mask == (False, True, False)
        await engine.aclose()

    asyncio.run(scenario())


def test_post_hook_can_reject_complete_groups() -> None:
    async def post_hook(group: RolloutGroup) -> RolloutGroup | None:
        return None if group.prompt.prompt_id == "reject" else group

    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(),
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 2, 2),
            post_hook,
        )
        await engine.rollout(
            prompts(
                RolloutPrompt("keep-1", (1,), {"target": 0}),
                RolloutPrompt("reject", (2,), {"target": 0}),
                RolloutPrompt("keep-2", (3,), {"target": 0}),
            )
        )
        batch = await engine.next_batch()
        assert {group.prompt.prompt_id for group in batch.groups} == {
            "keep-1",
            "keep-2",
        }
        await engine.aclose()

    asyncio.run(scenario())


def _invalid_hook(kind: str):
    async def hook(group: RolloutGroup) -> RolloutGroup:
        if kind == "count":
            return replace(group, samples=group.samples[:1])
        if kind == "version":
            version = group.policy_version + 1
            samples = tuple(
                replace(sample, policy_version=version) for sample in group.samples
            )
            return RolloutGroup(group.prompt, samples, version)
        if kind == "prompt":
            prompt = RolloutPrompt("changed", group.prompt.token_ids, group.prompt.metadata)
            samples = tuple(
                replace(sample, prompt_id=prompt.prompt_id) for sample in group.samples
            )
            return RolloutGroup(prompt, samples, group.policy_version)
        raise AssertionError(kind)

    return hook


@pytest.mark.parametrize("kind", ["count", "version", "prompt"])
def test_post_hook_output_is_strictly_validated(kind: str) -> None:
    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(),
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
            _invalid_hook(kind),
        )
        await engine.rollout(
            prompts(RolloutPrompt("p", (1,), {"target": 0}))
        )
        with pytest.raises(RolloutFailedError) as failed:
            await engine.next_batch()
        assert isinstance(failed.value.cause, InvalidRolloutGroupError)
        assert not isinstance(failed.value.cause, BaseExceptionGroup)
        await engine.aclose()

    asyncio.run(scenario())


def test_update_drains_groups_and_closes_admission_atomically() -> None:
    async def scenario() -> None:
        generation_release = asyncio.Event()
        old_group_started = asyncio.Event()
        update_started = asyncio.Event()
        update_release = asyncio.Event()
        old_request_count = 0

        async def control_generation(request) -> None:
            nonlocal old_request_count
            if request.prompt.prompt_id != "old":
                return
            old_request_count += 1
            if old_request_count == 2:
                old_group_started.set()
            await generation_release.wait()

        async def control_update(update: object, version: int) -> None:
            update_started.set()
            await update_release.wait()

        backend = FakeInferenceBackend(
            generate_control=control_generation,
            update_control=control_update,
        )
        engine = RolloutEngine(
            backend,
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )
        await engine.rollout(
            prompts(
                RolloutPrompt("old", (1,), {"target": 0}),
                RolloutPrompt("new", (2,), {"target": 0}),
            )
        )
        await old_group_started.wait()

        update_task = asyncio.create_task(engine.update_weights("weights-v1"))
        await asyncio.sleep(0)
        assert not update_started.is_set()

        generation_release.set()
        await update_started.wait()
        assert backend.updates == [("weights-v1", 1, 0)]
        assert {request.prompt.prompt_id for request in backend.requests} == {"old"}

        update_release.set()
        assert await update_task == 1
        await wait_until(lambda: len(backend.requests) == 4)

        first = await engine.next_batch()
        second = await engine.next_batch()
        by_prompt = {
            first.groups[0].prompt.prompt_id: first.groups[0],
            second.groups[0].prompt.prompt_id: second.groups[0],
        }
        assert by_prompt["old"].policy_version == 0
        assert by_prompt["new"].policy_version == 1
        await engine.aclose()

    asyncio.run(scenario())


def test_failed_weight_update_restores_admission_and_version() -> None:
    async def scenario() -> None:
        async def fail_update(update: object, version: int) -> None:
            raise RuntimeError("update failed")

        backend = FakeInferenceBackend(update_control=fail_update)
        engine = RolloutEngine(
            backend,
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )
        with pytest.raises(RuntimeError, match="update failed"):
            await engine.update_weights("bad")

        await engine.rollout(
            prompts(RolloutPrompt("p", (1,), {"target": 0}))
        )
        batch = await engine.next_batch()
        assert batch.groups[0].policy_version == 0
        await engine.aclose()

    asyncio.run(scenario())


def test_cancelled_weight_update_restores_admission_and_version() -> None:
    async def scenario() -> None:
        update_started = asyncio.Event()

        async def block_update(update: object, version: int) -> None:
            update_started.set()
            await asyncio.Event().wait()

        backend = FakeInferenceBackend(update_control=block_update)
        engine = RolloutEngine(
            backend,
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )

        update_task = asyncio.create_task(engine.update_weights("cancelled"))
        await update_started.wait()
        update_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update_task

        await engine.rollout(
            prompts(RolloutPrompt("p", (1,), {"target": 0}))
        )
        batch = await asyncio.wait_for(engine.next_batch(), timeout=1.0)
        assert batch.groups[0].policy_version == 0
        await engine.aclose()

    asyncio.run(scenario())


def test_source_exhaustion_reports_incomplete_batch() -> None:
    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(),
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 2, 1),
        )
        await engine.rollout(
            prompts(RolloutPrompt("only", (1,), {"target": 0}))
        )
        with pytest.raises(SourceExhaustedError) as exhausted:
            await engine.next_batch()
        assert len(exhausted.value.remaining_groups) == 1
        assert exhausted.value.remaining_groups[0].prompt.prompt_id == "only"
        await engine.aclose()

    asyncio.run(scenario())


def test_source_exception_is_propagated_to_batch_consumer() -> None:
    async def broken_source():
        raise RuntimeError("source failed")
        yield  # pragma: no cover

    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(),
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )
        await engine.rollout(broken_source())
        with pytest.raises(RolloutFailedError) as failed:
            await engine.next_batch()
        assert isinstance(failed.value.cause, RuntimeError)
        assert str(failed.value.cause) == "source failed"
        await engine.aclose()

    asyncio.run(scenario())


def test_workflow_exception_is_unwrapped_and_propagated() -> None:
    async def fail_one(request) -> None:
        if request.sample_index == 1:
            raise RuntimeError("generation failed")

    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(generate_control=fail_one),
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )
        await engine.rollout(
            prompts(RolloutPrompt("p", (1,), {"target": 0}))
        )
        with pytest.raises(RolloutFailedError) as failed:
            await engine.next_batch()
        assert isinstance(failed.value.cause, RuntimeError)
        assert not isinstance(failed.value.cause, BaseExceptionGroup)
        assert str(failed.value.cause) == "generation failed"
        await engine.aclose()

    asyncio.run(scenario())


def test_close_wakes_blocked_publishers_and_is_idempotent() -> None:
    async def many_prompts():
        for index in range(20):
            yield RolloutPrompt(str(index), (index,), {"target": 0})

    async def scenario() -> None:
        backend = FakeInferenceBackend()
        engine = RolloutEngine(
            backend,
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 2, prefetch_batches=1),
        )
        await engine.rollout(many_prompts())
        await wait_until(lambda: len(backend.requests) >= 6)

        await asyncio.wait_for(engine.aclose(), timeout=1.0)
        assert backend.closed_count == 1
        await engine.aclose()
        assert backend.closed_count == 1
        with pytest.raises(EngineClosedError):
            await engine.next_batch()

    asyncio.run(scenario())


def test_rollout_rejects_a_second_prompt_source() -> None:
    async def scenario() -> None:
        engine = RolloutEngine(
            FakeInferenceBackend(),
            SingleTurnWorkflow(),
            metadata_reward,
            RolloutConfig(2, 1, 1),
        )
        await engine.rollout(prompts())
        with pytest.raises(RolloutAlreadyStartedError):
            await engine.rollout(prompts())
        await engine.aclose()

    asyncio.run(scenario())
