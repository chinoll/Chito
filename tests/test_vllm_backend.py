from __future__ import annotations

import asyncio
import builtins
import sys
import threading
from dataclasses import dataclass
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


class FakeWeightTransferInitRequest:
    def __init__(self, *, init_info) -> None:
        self.init_info = init_info


class FakeWeightTransferUpdateRequest:
    def __init__(self, *, update_info) -> None:
        self.update_info = update_info


class FakeNCCLSendArgs:
    def __init__(self, *, group, packed) -> None:
        self.group = group
        self.packed = packed


class FakeNCCLGroup:
    def __init__(self) -> None:
        self.destroy_count = 0

    def destroy(self) -> None:
        self.destroy_count += 1


class FakeNCCLEngine:
    group = FakeNCCLGroup()
    init_started = threading.Event()
    send_started = threading.Event()
    send_release = threading.Event()
    init_calls: list[dict[str, object]] = []
    send_calls: list[tuple[tuple[tuple[str, object], ...], FakeNCCLSendArgs]] = []

    @classmethod
    def reset(cls) -> None:
        cls.group = FakeNCCLGroup()
        cls.init_started = threading.Event()
        cls.send_started = threading.Event()
        cls.send_release = threading.Event()
        cls.send_release.set()
        cls.init_calls = []
        cls.send_calls = []

    @classmethod
    def trainer_init(cls, init_info):
        cls.init_calls.append(init_info)
        cls.init_started.set()
        return cls.group

    @classmethod
    def trainer_send_weights(cls, iterator, args) -> None:
        cls.send_calls.append((tuple(iterator), args))
        cls.send_started.set()
        assert cls.send_release.wait(1)


class FakeCudaDeviceContext:
    def __init__(self, device) -> None:
        self.device = device

    def __enter__(self) -> None:
        FakeCuda.devices.append(self.device)

    def __exit__(self, *_exc) -> None:
        FakeCuda.restored += 1


class FakeCuda:
    devices: list[object] = []
    restored = 0

    @classmethod
    def device(cls, device) -> FakeCudaDeviceContext:
        return FakeCudaDeviceContext(device)


@dataclass(frozen=True)
class FakeDevice:
    type: str = "cuda"
    index: int = 0


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str = "torch.float16"
    device: FakeDevice = FakeDevice()


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
        self.weight_init_requests: list[FakeWeightTransferInitRequest] = []
        self.weight_update_requests: list[FakeWeightTransferUpdateRequest] = []

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

    async def init_weight_transfer_engine(self, request) -> None:
        self.weight_init_requests.append(request)
        self._phase("init")
        initialized = await asyncio.to_thread(FakeNCCLEngine.init_started.wait, 1)
        assert initialized

    async def start_weight_update(self, *, is_checkpoint_format) -> None:
        assert is_checkpoint_format is True
        self._phase("start")

    async def update_weights(self, request) -> None:
        self.weight_update_requests.append(request)
        self._phase("receive")
        sent = await asyncio.to_thread(FakeNCCLEngine.send_started.wait, 1)
        assert sent

    async def finish_weight_update(self) -> None:
        self._phase("finish")

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
    FakeNCCLEngine.reset()
    FakeCuda.devices = []
    FakeCuda.restored = 0

    module = ModuleType("vllm")
    module.AsyncEngineArgs = FakeAsyncEngineArgs
    module.AsyncLLMEngine = FakeAsyncLLMEngine
    module.SamplingParams = FakeSamplingParams
    module.TokensPrompt = lambda **kwargs: kwargs
    base_module = ModuleType("vllm.distributed.weight_transfer.base")
    base_module.WeightTransferInitRequest = FakeWeightTransferInitRequest
    base_module.WeightTransferUpdateRequest = FakeWeightTransferUpdateRequest
    nccl_module = ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    nccl_module.NCCLTrainerSendWeightsArgs = FakeNCCLSendArgs
    nccl_module.NCCLWeightTransferEngine = FakeNCCLEngine
    torch_module = ModuleType("torch")
    torch_module.cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", ModuleType("vllm.distributed"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.weight_transfer",
        ModuleType("vllm.distributed.weight_transfer"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.weight_transfer.base", base_module
    )
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.weight_transfer.nccl_engine", nccl_module
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
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
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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


def test_backend_broadcasts_nccl_weights_and_reuses_group(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend(
            "model-id",
            engine_kwargs={
                "data_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "tensor_parallel_size": 2,
            },
        )
        fake_engine = FakeAsyncLLMEngine.instance
        update = VllmNcclWeightUpdate(
            [
                ("model.a", FakeTensor((2, 3))),
                ("model.b", FakeTensor((4,), dtype="torch.float32")),
            ]
        )

        await backend.update_weights(update, new_policy_version=1)
        await backend.update_weights(update, new_policy_version=2)

        assert len(FakeNCCLEngine.init_calls) == 1
        init_info = fake_engine.weight_init_requests[0].init_info
        assert init_info["master_address"] == "127.0.0.1"
        assert init_info["rank_offset"] == 1
        assert init_info["world_size"] == 5
        assert FakeNCCLEngine.init_calls == [
            {
                "master_address": "127.0.0.1",
                "master_port": init_info["master_port"],
                "world_size": 5,
            }
        ]
        assert [request.update_info for request in fake_engine.weight_update_requests] == [
            {
                "names": ["model.a", "model.b"],
                "dtype_names": ["float16", "float32"],
                "shapes": [[2, 3], [4]],
                "packed": False,
            },
            {
                "names": ["model.a", "model.b"],
                "dtype_names": ["float16", "float32"],
                "shapes": [[2, 3], [4]],
                "packed": False,
            },
        ]
        assert len(FakeNCCLEngine.send_calls) == 2
        assert all(
            call[1].group is FakeNCCLEngine.group and call[1].packed is False
            for call in FakeNCCLEngine.send_calls
        )
        assert fake_engine.events == [
            "pause",
            "init",
            "start",
            "receive",
            "finish",
            "resume",
            "pause",
            "start",
            "receive",
            "finish",
            "resume",
        ]
        await backend.aclose()
        assert FakeNCCLEngine.group.destroy_count == 1
        assert FakeCuda.devices == [FakeDevice(), FakeDevice(), FakeDevice()]
        assert FakeCuda.restored == 3

    asyncio.run(scenario())


def test_nccl_failure_waits_for_sender_before_poisoning(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures["receive"] = 1
        FakeNCCLEngine.send_release.clear()
        update = VllmNcclWeightUpdate([("model.a", FakeTensor((1,)))])

        update_task = asyncio.create_task(
            backend.update_weights(update, new_policy_version=1)
        )
        sent = await asyncio.to_thread(FakeNCCLEngine.send_started.wait, 1)
        assert sent
        await asyncio.sleep(0)
        assert not update_task.done()

        generation_task = asyncio.create_task(backend.generate(request()))
        await asyncio.sleep(0)
        assert not generation_task.done()

        FakeNCCLEngine.send_release.set()
        with pytest.raises(RuntimeError, match="receive failed"):
            await update_task
        with pytest.raises(VllmBackendPoisonedError):
            await generation_task
        assert fake_engine.events == ["pause", "init", "start", "receive"]
        await backend.aclose()
        assert FakeNCCLEngine.group.destroy_count == 1

    asyncio.run(scenario())


def test_failed_nccl_init_destroys_trainer_group(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        fake_engine = FakeAsyncLLMEngine.instance
        fake_engine.failures["init"] = 1
        update = VllmNcclWeightUpdate([("model.a", FakeTensor((1,)))])

        with pytest.raises(RuntimeError, match="init failed"):
            await backend.update_weights(update, new_policy_version=1)

        assert FakeNCCLEngine.group.destroy_count == 1
        await backend.aclose()
        assert FakeNCCLEngine.group.destroy_count == 1

    asyncio.run(scenario())


def test_transfer_mode_requires_matching_update_type(fake_vllm) -> None:
    async def scenario() -> None:
        nccl_backend = VllmBackend("model-id")
        with pytest.raises(TypeError, match="nccl transfer requires VllmNccl"):
            await nccl_backend.update_weights(weight_update(), new_policy_version=1)
        await nccl_backend.aclose()

        checkpoint_backend = VllmBackend(
            "model-id", weight_transfer="checkpoint"
        )
        with pytest.raises(TypeError, match="checkpoint transfer requires"):
            await checkpoint_backend.update_weights(
                VllmNcclWeightUpdate([("model.a", FakeTensor((1,)))]),
                new_policy_version=1,
            )
        await checkpoint_backend.aclose()

    asyncio.run(scenario())


def test_invalid_checkpoint_before_admission_can_be_retried(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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
        assert fake_engine.events == ["pause", "reload", "resume"]
        await backend.aclose()

    asyncio.run(scenario())


def test_backend_rejects_unknown_weight_update_type(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id", weight_transfer="checkpoint")
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
            {"engine_kwargs": {"nnodes": 2}},
            "currently requires one node",
        ),
        (
            {"engine_kwargs": {"prefill_context_parallel_size": 2}},
            "does not support prefill context parallelism",
        ),
        (
            {"engine_kwargs": {"distributed_executor_backend": "external_launcher"}},
            "does not support external_launcher",
        ),
        (
            {"engine_kwargs": {"weight_transfer_config": {}}},
            "pass weight_transfer directly",
        ),
    ],
)
def test_backend_configuration_fails_fast(fake_vllm, kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        VllmBackend("model-id", **kwargs)


def test_nccl_rejects_vllm_data_parallel_environment(fake_vllm, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DP_SIZE", "2")

    with pytest.raises(ValueError, match="pass data_parallel_size"):
        VllmBackend("model-id")


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
