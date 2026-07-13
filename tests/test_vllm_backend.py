from __future__ import annotations

import asyncio
import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from chito import (
    InferenceRequest,
    RolloutPrompt,
    VllmBackend,
    VllmBackendPoisonedError,
    VllmCheckpointWeightUpdate,
    VllmNcclWeightUpdate,
)


class FakeAsyncEngineArgs:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeAsyncLLMEngine:
    instance: FakeAsyncLLMEngine
    checkpoint_path: Path

    def __init__(self, engine_args: FakeAsyncEngineArgs) -> None:
        self.engine_args = engine_args
        self.calls: list[tuple[dict[str, object], FakeSamplingParams, str]] = []
        self.active = 0
        self.max_active = 0
        self.shutdown_count = 0
        self.two_requests_started = asyncio.Event()
        self.release = asyncio.Event()
        self.events: list[str] = []
        self.failures: dict[str, int] = {}
        self.reload_calls: list[tuple[str, dict[str, object]]] = []
        self.reload_started = asyncio.Event()
        self.reload_release = asyncio.Event()
        self.reload_release.set()

    @classmethod
    def from_engine_args(
        cls, engine_args: FakeAsyncEngineArgs
    ) -> FakeAsyncLLMEngine:
        cls.instance = cls(engine_args)
        return cls.instance

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append((prompt, sampling_params, request_id))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 2:
            self.two_requests_started.set()
        await self.release.wait()
        try:
            output_token = 100 + len(self.calls)
            completion = SimpleNamespace(
                token_ids=[output_token, 200],
                logprobs=[
                    {output_token: SimpleNamespace(logprob=-0.25)},
                    {200: SimpleNamespace(logprob=-0.5)},
                ],
            )
            yield SimpleNamespace(
                finished=True,
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[completion],
            )
        finally:
            self.active -= 1

    async def pause_generation(self, *, mode, clear_cache) -> None:
        assert mode == "wait"
        assert clear_cache is True
        self._phase("pause")

    async def collective_rpc(self, method, *, kwargs) -> None:
        self.reload_calls.append((method, kwargs))
        self._phase("reload")
        self.reload_started.set()
        await self.reload_release.wait()

    async def resume_generation(self) -> None:
        self._phase("resume")

    def _phase(self, name: str) -> None:
        self.events.append(name)
        remaining = self.failures.get(name, 0)
        if remaining:
            self.failures[name] = remaining - 1
            raise RuntimeError(f"{name} failed")

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def fake_vllm(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    FakeAsyncLLMEngine.checkpoint_path = checkpoint

    module = ModuleType("vllm")
    module.AsyncEngineArgs = FakeAsyncEngineArgs
    module.AsyncLLMEngine = FakeAsyncLLMEngine
    module.SamplingParams = FakeSamplingParams
    module.TokensPrompt = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def request(sample_index: int = 0) -> InferenceRequest:
    return InferenceRequest(
        prompt=RolloutPrompt("prompt", (11, 12, 13)),
        sample_index=sample_index,
        policy_version=7,
    )


def weight_update() -> VllmCheckpointWeightUpdate:
    return VllmCheckpointWeightUpdate(FakeAsyncLLMEngine.checkpoint_path)


def test_backend_uses_token_prompt_and_sampled_logprobs(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend(
            "model-id",
            max_tokens=9,
            temperature=0.7,
            top_p=0.8,
            engine_kwargs={"dtype": "float16"},
        )
        fake_engine = FakeAsyncLLMEngine.instance

        task_a = asyncio.create_task(backend.generate(request(0)))
        task_b = asyncio.create_task(backend.generate(request(1)))
        await fake_engine.two_requests_started.wait()
        fake_engine.release.set()
        result_a, result_b = await asyncio.gather(task_a, task_b)

        assert fake_engine.engine_args.kwargs == {
            "model": "model-id",
            "dtype": "float16",
            "weight_transfer_config": {"backend": "nccl"},
        }
        assert fake_engine.max_active == 2
        assert [call[0] for call in fake_engine.calls] == [
            {"prompt_token_ids": [11, 12, 13]},
            {"prompt_token_ids": [11, 12, 13]},
        ]
        assert len({call[2] for call in fake_engine.calls}) == 2
        assert fake_engine.calls[0][1].kwargs == {
            "n": 1,
            "max_tokens": 9,
            "temperature": 0.7,
            "top_p": 0.8,
            "logprobs": 0,
            "detokenize": False,
        }
        for result in (result_a, result_b):
            assert result.output_logprobs == (-0.25, -0.5)
            assert result.policy_version == 7

        await backend.aclose()
        await backend.aclose()
        assert fake_engine.shutdown_count == 1

    asyncio.run(scenario())


def test_backend_reloads_complete_local_checkpoint(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        update = weight_update()

        await backend.update_weights(update, new_policy_version=1)

        assert fake_engine.events == ["pause", "reload", "resume"]
        assert fake_engine.reload_calls == [
            (
                "reload_weights",
                {
                    "weights_path": str(update.checkpoint_path),
                    "is_checkpoint_format": True,
                },
            )
        ]
        await backend.aclose()

    asyncio.run(scenario())


def test_checkpoint_transfer_does_not_configure_nccl(fake_vllm) -> None:
    VllmBackend("model-id", weight_transfer="checkpoint")

    assert FakeAsyncLLMEngine.instance.engine_args.kwargs == {"model": "model-id"}


def test_nccl_weight_update_freezes_named_weights() -> None:
    update = VllmNcclWeightUpdate(iter([("layer.weight", object())]))

    assert isinstance(update.named_weights, tuple)
    assert update.named_weights[0][0] == "layer.weight"
    with pytest.raises(ValueError, match="must not be empty"):
        VllmNcclWeightUpdate(iter(()))


def test_invalid_checkpoint_before_admission_can_be_retried(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        update = weight_update()
        update.checkpoint_path.rmdir()

        with pytest.raises(FileNotFoundError):
            await backend.update_weights(update, new_policy_version=1)
        assert fake_engine.events == []

        update.checkpoint_path.mkdir()
        await backend.update_weights(update, new_policy_version=1)
        assert fake_engine.events == ["pause", "reload", "resume"]
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["pause", "reload", "resume"])
def test_unsafe_update_failure_poisons_backend(fake_vllm, phase: str) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures[phase] = 1

        with pytest.raises(RuntimeError, match=f"{phase} failed"):
            await backend.update_weights(weight_update(), new_policy_version=1)

        with pytest.raises(VllmBackendPoisonedError) as poisoned_update:
            await backend.update_weights(weight_update(), new_policy_version=1)
        assert phase in str(poisoned_update.value.cause)

        with pytest.raises(VllmBackendPoisonedError):
            await backend.generate(request())
        await backend.aclose()

    asyncio.run(scenario())


def test_update_drains_requests_and_blocks_new_generation(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance

        first_generation = asyncio.create_task(backend.generate(request(0)))
        while fake_engine.active != 1:
            await asyncio.sleep(0)

        update_task = asyncio.create_task(
            backend.update_weights(weight_update(), new_policy_version=1)
        )
        await asyncio.sleep(0)
        second_generation = asyncio.create_task(backend.generate(request(1)))
        for _ in range(10):
            await asyncio.sleep(0)

        assert len(fake_engine.calls) == 1
        assert fake_engine.events == []

        fake_engine.release.set()
        await first_generation
        await update_task
        await second_generation

        assert len(fake_engine.calls) == 2
        assert fake_engine.events == ["pause", "reload", "resume"]
        await backend.aclose()

    asyncio.run(scenario())


def test_cancelled_update_waiter_reopens_generation(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance

        first_generation = asyncio.create_task(backend.generate(request(0)))
        while fake_engine.active != 1:
            await asyncio.sleep(0)

        update_task = asyncio.create_task(
            backend.update_weights(weight_update(), new_policy_version=1)
        )
        await asyncio.sleep(0)
        update_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update_task

        second_generation = asyncio.create_task(backend.generate(request(1)))
        await fake_engine.two_requests_started.wait()
        assert fake_engine.events == []

        fake_engine.release.set()
        await asyncio.gather(first_generation, second_generation)
        await backend.aclose()

    asyncio.run(scenario())


def test_cancelled_active_reload_finishes_before_backend_is_poisoned(
    fake_vllm,
) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.reload_release.clear()

        update_task = asyncio.create_task(
            backend.update_weights(weight_update(), new_policy_version=1)
        )
        await fake_engine.reload_started.wait()
        update_task.cancel()
        await asyncio.sleep(0)
        assert not update_task.done()

        generation_task = asyncio.create_task(backend.generate(request()))
        await asyncio.sleep(0)
        assert not generation_task.done()

        fake_engine.reload_release.set()
        with pytest.raises(asyncio.CancelledError):
            await update_task
        with pytest.raises(VllmBackendPoisonedError) as poisoned:
            await generation_task
        assert isinstance(poisoned.value.cause, asyncio.CancelledError)
        assert fake_engine.events == ["pause", "reload"]
        await backend.aclose()

    asyncio.run(scenario())


def test_backend_rejects_unknown_weight_update_type(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        with pytest.raises(TypeError, match="VllmCheckpointWeightUpdate"):
            await backend.update_weights(object(), new_policy_version=1)
        await backend.aclose()

    asyncio.run(scenario())


def test_chito_import_does_not_import_optional_vllm(monkeypatch) -> None:
    imported_vllm = False
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        nonlocal imported_vllm
        if name == "vllm" or name.startswith("vllm."):
            imported_vllm = True
            raise AssertionError("optional vllm import was attempted")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("chito", None)
    imported = __import__("chito")

    assert imported.VllmBackend is not None
    assert not imported_vllm


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"temperature": -0.1}, "temperature must be non-negative"),
        ({"top_p": 0.0}, "top_p must be in the interval"),
        ({"engine_kwargs": {"model": "other"}}, "pass model directly"),
        ({"weight_transfer": "other"}, "weight_transfer must be"),
        (
            {"engine_kwargs": {"weight_transfer_config": {}}},
            "pass weight_transfer directly",
        ),
    ],
)
def test_backend_configuration_fails_fast(fake_vllm, kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        VllmBackend("model-id", **kwargs)


def test_weight_update_requires_existing_directory(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    update = VllmCheckpointWeightUpdate(checkpoint)

    assert update.checkpoint_path == checkpoint.resolve()
    with pytest.raises(ValueError, match="must not be empty"):
        VllmCheckpointWeightUpdate("")
    with pytest.raises(FileNotFoundError):
        VllmCheckpointWeightUpdate(tmp_path / "missing")

    checkpoint_file = tmp_path / "checkpoint.bin"
    checkpoint_file.touch()
    with pytest.raises(NotADirectoryError):
        VllmCheckpointWeightUpdate(checkpoint_file)
