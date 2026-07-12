from __future__ import annotations

import asyncio
import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from chito import InferenceRequest, RolloutPrompt, VllmBackend


class FakeAsyncEngineArgs:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


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

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def fake_vllm(monkeypatch):
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


def test_backend_rejects_dynamic_weight_updates(fake_vllm) -> None:
    async def scenario() -> None:
        backend = VllmBackend("model-id")
        with pytest.raises(NotImplementedError, match="dynamic weight updates"):
            await backend.update_weights(object(), new_policy_version=1)
        FakeAsyncLLMEngine.instance.release.set()
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
    ],
)
def test_backend_configuration_fails_fast(fake_vllm, kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        VllmBackend("model-id", **kwargs)
