from __future__ import annotations

import asyncio
import builtins
import sys
import threading
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from chito import (
    InferenceRequest,
    RolloutPrompt,
    VllmBackend,
    VllmBackendPoisonedError,
    VllmWeightUpdate,
)


class FakeAsyncEngineArgs:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@dataclass
class FakeWeightTransferInitRequest:
    init_info: dict[str, object]


@dataclass
class FakeWeightTransferUpdateRequest:
    update_info: dict[str, object]


@dataclass
class FakeIPCUpdateInfo:
    names: list[str]
    packed: bool


@dataclass
class FakeIPCTrainerSendWeightsArgs:
    send_mode: object
    packed: bool = False
    packed_buffer_size_bytes: int = 1 << 30


class FakeAsyncLLMEngine:
    instance: FakeAsyncLLMEngine

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
        self.sender_args: list[FakeIPCTrainerSendWeightsArgs] = []
        self.sender_weights: list[tuple[tuple[str, object], ...]] = []
        self.sender_thread_ids: list[int] = []
        self.receiver_thread_ids: list[int] = []
        self.update_requests: list[FakeWeightTransferUpdateRequest] = []

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

    async def resume_generation(self) -> None:
        self._phase("resume")

    async def init_weight_transfer_engine(self, request) -> None:
        assert request.init_info == {}
        self._phase("init")

    async def start_weight_update(self, *, is_checkpoint_format) -> None:
        self.events.append(f"start:{is_checkpoint_format}")
        self._fail_if_requested("start")

    async def update_weights(self, request) -> None:
        self.receiver_thread_ids.append(threading.get_ident())
        self.update_requests.append(request)
        self._phase("receive")

    async def finish_weight_update(self) -> None:
        self._phase("finish")

    def _phase(self, name: str) -> None:
        self.events.append(name)
        self._fail_if_requested(name)

    def _fail_if_requested(self, name: str) -> None:
        remaining = self.failures.get(name, 0)
        if remaining:
            self.failures[name] = remaining - 1
            raise RuntimeError(f"{name} failed")

    def shutdown(self) -> None:
        self.shutdown_count += 1


class FakeIPCWeightTransferEngine:
    @staticmethod
    def trainer_send_weights(iterator, args) -> None:
        engine = FakeAsyncLLMEngine.instance
        engine.events.append("send")
        engine.sender_thread_ids.append(threading.get_ident())
        engine.sender_args.append(args)
        weights = tuple(iterator)
        engine.sender_weights.append(weights)
        engine._fail_if_requested("send")
        args.send_mode(
            FakeIPCUpdateInfo(
                names=[name for name, _tensor in weights],
                packed=args.packed,
            )
        )


@pytest.fixture
def fake_vllm(monkeypatch):
    module = ModuleType("vllm")
    module.__path__ = []
    module.AsyncEngineArgs = FakeAsyncEngineArgs
    module.AsyncLLMEngine = FakeAsyncLLMEngine
    module.SamplingParams = FakeSamplingParams
    module.TokensPrompt = lambda **kwargs: kwargs

    distributed = ModuleType("vllm.distributed")
    distributed.__path__ = []
    weight_transfer = ModuleType("vllm.distributed.weight_transfer")
    weight_transfer.__path__ = []
    base = ModuleType("vllm.distributed.weight_transfer.base")
    base.WeightTransferInitRequest = FakeWeightTransferInitRequest
    base.WeightTransferUpdateRequest = FakeWeightTransferUpdateRequest
    ipc = ModuleType("vllm.distributed.weight_transfer.ipc_engine")
    ipc.IPCTrainerSendWeightsArgs = FakeIPCTrainerSendWeightsArgs
    ipc.IPCWeightTransferEngine = FakeIPCWeightTransferEngine

    modules = {
        "vllm": module,
        "vllm.distributed": distributed,
        "vllm.distributed.weight_transfer": weight_transfer,
        "vllm.distributed.weight_transfer.base": base,
        "vllm.distributed.weight_transfer.ipc_engine": ipc,
    }
    for name, fake_module in modules.items():
        monkeypatch.setitem(sys.modules, name, fake_module)
    return module


def request(sample_index: int = 0) -> InferenceRequest:
    return InferenceRequest(
        prompt=RolloutPrompt("prompt", (11, 12, 13)),
        sample_index=sample_index,
        policy_version=7,
    )


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
            "weight_transfer_config": {"backend": "ipc"},
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


def weight_update(**kwargs) -> VllmWeightUpdate:
    return VllmWeightUpdate(
        (("layer.weight", object()), ("layer.bias", object())),
        **kwargs,
    )


def test_backend_rejects_unknown_weight_update_type(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        with pytest.raises(TypeError, match="VllmWeightUpdate"):
            await backend.update_weights(object(), new_policy_version=1)
        await backend.aclose()

    asyncio.run(scenario())


def test_backend_runs_four_phase_weight_update_on_thread(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        loop_thread_id = threading.get_ident()
        update = weight_update(
            checkpoint_format=False,
            packed=True,
            packed_buffer_size_bytes=4096,
        )

        await backend.update_weights(update, new_policy_version=1)

        assert fake_engine.events == [
            "pause",
            "init",
            "start:False",
            "send",
            "receive",
            "finish",
            "resume",
        ]
        assert fake_engine.sender_weights == [update.weights]
        assert fake_engine.sender_args[0].packed is True
        assert fake_engine.sender_args[0].packed_buffer_size_bytes == 4096
        assert fake_engine.sender_thread_ids[0] != loop_thread_id
        assert fake_engine.receiver_thread_ids == [loop_thread_id]
        assert fake_engine.update_requests[0].update_info == {
            "names": ["layer.weight", "layer.bias"],
            "packed": True,
        }

        fake_engine.events.clear()
        await backend.update_weights(weight_update(), new_policy_version=2)
        assert fake_engine.events == [
            "pause",
            "start:True",
            "send",
            "receive",
            "finish",
            "resume",
        ]
        await backend.aclose()

    asyncio.run(scenario())


def test_initialization_failure_resumes_and_can_retry(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures["init"] = 1

        with pytest.raises(RuntimeError, match="init failed"):
            await backend.update_weights(weight_update(), new_policy_version=1)
        assert fake_engine.events == ["pause", "init", "resume"]

        fake_engine.events.clear()
        await backend.update_weights(weight_update(), new_policy_version=1)
        assert fake_engine.events == [
            "pause",
            "init",
            "start:True",
            "send",
            "receive",
            "finish",
            "resume",
        ]
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "phase", ["pause", "start", "send", "finish", "resume"]
)
def test_unsafe_update_failure_poisons_backend(fake_vllm, phase: str) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures[phase] = 1

        with pytest.raises(RuntimeError, match=f"{phase} failed"):
            await backend.update_weights(weight_update(), new_policy_version=1)

        if phase == "send":
            assert fake_engine.events[-1] == "finish"
        assert "resume" not in fake_engine.events or phase == "resume"

        with pytest.raises(VllmBackendPoisonedError) as poisoned_update:
            await backend.update_weights(weight_update(), new_policy_version=1)
        assert phase in str(poisoned_update.value.cause)

        with pytest.raises(VllmBackendPoisonedError):
            await backend.generate(request())
        await backend.aclose()

    asyncio.run(scenario())


def test_update_preserves_transfer_and_cleanup_errors(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures.update({"send": 1, "finish": 1})

        with pytest.raises(BaseExceptionGroup) as failed:
            await backend.update_weights(weight_update(), new_policy_version=1)
        assert [str(error) for error in failed.value.exceptions] == [
            "send failed",
            "finish failed",
        ]

        with pytest.raises(VllmBackendPoisonedError):
            await backend.update_weights(weight_update(), new_policy_version=1)
        await backend.aclose()

    asyncio.run(scenario())


def test_initialization_and_resume_failure_poisons_backend(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures.update({"init": 1, "resume": 1})

        with pytest.raises(BaseExceptionGroup) as failed:
            await backend.update_weights(weight_update(), new_policy_version=1)
        assert [str(error) for error in failed.value.exceptions] == [
            "init failed",
            "resume failed",
        ]

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
        assert fake_engine.events == [
            "pause",
            "init",
            "start:True",
            "send",
            "receive",
            "finish",
            "resume",
        ]
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
        (
            {"engine_kwargs": {"weight_transfer_config": {"backend": "nccl"}}},
            "manages weight_transfer_config",
        ),
    ],
)
def test_backend_configuration_fails_fast(fake_vllm, kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        VllmBackend("model-id", **kwargs)


def test_weight_update_materializes_and_validates_weights() -> None:
    update = VllmWeightUpdate(
        ((name, object()) for name in ("a", "b")),
        checkpoint_format=False,
        packed=True,
        packed_buffer_size_bytes=1024,
    )

    assert isinstance(update.weights, tuple)
    assert [name for name, _tensor in update.weights] == ["a", "b"]
    assert not update.checkpoint_format
    assert update.packed
    assert update.packed_buffer_size_bytes == 1024


@pytest.mark.parametrize(
    ("weights", "kwargs", "error"),
    [
        ([], {}, "at least one"),
        ([("", object())], {}, "non-empty"),
        ([("a", object()), ("a", object())], {}, "duplicate"),
        ([("a", object(), object())], {}, r"\(name, tensor\)"),
        ([("a", object())], {"packed_buffer_size_bytes": 0}, "positive"),
    ],
)
def test_weight_update_configuration_fails_fast(
    weights, kwargs, error
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        VllmWeightUpdate(weights, **kwargs)
