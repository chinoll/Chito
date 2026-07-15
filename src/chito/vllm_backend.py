"""vLLM generation and the receiver half of NCCL weight updates."""

from __future__ import annotations

import asyncio
import itertools
import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .models import InferenceRequest, InferenceResult


class VllmBackendPoisonedError(RuntimeError):
    """A partial weight update made the loaded inference model unsafe."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(
            "vLLM failed during a weight update; close the rollout service"
        )


class VllmBackend:
    """Own vLLM inside the rollout service process."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        engine_kwargs: Mapping[str, object],
    ) -> None:
        options = dict(engine_kwargs)
        options["logprobs_mode"] = "processed_logprobs"
        options["weight_transfer_config"] = {"backend": "nccl"}

        try:
            from vllm import (
                AsyncEngineArgs,
                AsyncLLMEngine,
                SamplingParams,
                TokensPrompt,
            )
            from vllm.distributed.weight_transfer.base import (
                WeightTransferInitRequest,
                WeightTransferUpdateRequest,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "vllm":
                raise
            raise ModuleNotFoundError(
                "the rollout service requires chito[vllm]"
            ) from exc

        engine_args = AsyncEngineArgs(model=model, **options)
        agent_store = os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        try:
            self._engine: Any = AsyncLLMEngine.from_engine_args(engine_args)
        finally:
            if agent_store is not None:
                os.environ["TORCHELASTIC_USE_AGENT_STORE"] = agent_store

        self._sampling_params: Any = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            logprobs=0,
            detokenize=False,
        )
        self._tokens_prompt = TokensPrompt
        self._init_request = WeightTransferInitRequest
        self._update_request = WeightTransferUpdateRequest
        self._inference_world_size = (
            int(options.get("data_parallel_size", 1))
            * int(options.get("pipeline_parallel_size", 1))
            * int(options.get("tensor_parallel_size", 1))
        )

        self._request_prefix = f"chito-{uuid.uuid4().hex}"
        self._request_ids = itertools.count()
        self._condition = asyncio.Condition()
        self._active_requests = 0
        self._updating = False
        self._channel_initialized = False
        self._receive_task: asyncio.Task[Any] | None = None
        self._poisoned: BaseException | None = None
        self._closing = False
        self._closed = False

    @property
    def inference_world_size(self) -> int:
        return self._inference_world_size

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate from exact token IDs and keep sampled-token logprobs."""
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")

        await self._begin_request()
        try:
            prompt = self._tokens_prompt(
                prompt_token_ids=list(request.prompt.token_ids)
            )
            request_id = f"{self._request_prefix}-{next(self._request_ids)}"
            final_output = None
            async for output in self._engine.generate(
                prompt, self._sampling_params, request_id
            ):
                final_output = output
            return self._to_result(final_output, request)
        finally:
            await self._finish_request()

    async def init_weight_channel(self, init_info: Mapping[str, object]) -> None:
        """Join the persistent NCCL communicator as the vLLM receivers."""
        if self._channel_initialized:
            raise RuntimeError("weight channel is already initialized")
        request = self._init_request(init_info=dict(init_info))
        await self._engine.init_weight_transfer_engine(request)
        self._channel_initialized = True

    async def begin_weight_update(
        self, update_info: Mapping[str, object]
    ) -> asyncio.Task[Any]:
        """Pause generation and start vLLM's asynchronous NCCL receive."""
        if not self._channel_initialized:
            raise RuntimeError("weight channel is not initialized")
        await self._begin_update()
        try:
            await self._engine.pause_generation(mode="wait", clear_cache=True)
            await self._engine.start_weight_update()
            request = self._update_request(update_info=dict(update_info))
            task = asyncio.create_task(self._engine.update_weights(request))
            self._receive_task = task
            return task
        except BaseException as exc:
            await self._end_update(exc)
            raise

    async def finish_weight_update(self, receive_task: asyncio.Task[Any]) -> None:
        """Commit a completed receive and resume generation."""
        if receive_task is not self._receive_task:
            raise RuntimeError("receive task does not match the pending update")
        failure: BaseException | None = None
        try:
            await receive_task
            await self._engine.finish_weight_update()
            await self._engine.resume_generation()
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self._receive_task = None
            await self._end_update(failure)

    async def fail_weight_update(self, message: str) -> None:
        """Poison the backend after the trainer sender failed."""
        failure = RuntimeError(message)
        task = self._receive_task
        if task is not None and not task.done():
            task.cancel()
        self._receive_task = None
        await self._end_update(failure)

    async def aclose(self) -> None:
        """Stop admitted work and release the vLLM engine."""
        async with self._condition:
            if self._closed:
                return
            self._closing = True
            self._condition.notify_all()
            await self._condition.wait_for(lambda: self._active_requests == 0)

        task = self._receive_task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        try:
            self._engine.shutdown()
        finally:
            async with self._condition:
                self._closed = True
                self._condition.notify_all()

    async def _begin_request(self) -> None:
        async with self._condition:
            while self._updating:
                self._raise_if_unusable()
                await self._condition.wait()
            self._raise_if_unusable()
            self._active_requests += 1

    async def _finish_request(self) -> None:
        async with self._condition:
            self._active_requests -= 1
            self._condition.notify_all()

    async def _begin_update(self) -> None:
        async with self._condition:
            self._raise_if_unusable()
            if self._updating:
                raise RuntimeError("a weight update is already running")
            self._updating = True
            await self._condition.wait_for(lambda: self._active_requests == 0)

    async def _end_update(self, failure: BaseException | None) -> None:
        async with self._condition:
            if failure is not None and self._poisoned is None:
                self._poisoned = failure
            self._updating = False
            self._condition.notify_all()

    def _raise_if_unusable(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("vLLM backend is closed")
        if self._poisoned is not None:
            raise VllmBackendPoisonedError(self._poisoned) from self._poisoned

    @staticmethod
    def _to_result(output: Any, request: InferenceRequest) -> InferenceResult:
        if output is None or not output.finished:
            raise RuntimeError("vLLM generation ended without a final output")
        if tuple(output.prompt_token_ids) != request.prompt.token_ids:
            raise RuntimeError("vLLM changed the supplied prompt token IDs")
        if len(output.outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected completion count")

        completion = output.outputs[0]
        token_ids = tuple(int(token_id) for token_id in completion.token_ids)
        logprobs = VllmBackend._sampled_logprobs(token_ids, completion.logprobs)
        return InferenceResult(
            output_token_ids=token_ids,
            output_logprobs=logprobs,
            policy_version=request.policy_version,
        )

    @staticmethod
    def _sampled_logprobs(
        token_ids: tuple[int, ...],
        positions: Sequence[Mapping[int, Any]] | None,
    ) -> tuple[float, ...]:
        if positions is None or len(positions) != len(token_ids):
            raise RuntimeError("vLLM returned incomplete sampled-token logprobs")

        values: list[float] = []
        for token_id, candidates in zip(token_ids, positions, strict=True):
            sampled = candidates.get(token_id)
            if sampled is None:
                raise RuntimeError("vLLM omitted a sampled token logprob")
            values.append(float(sampled.logprob))
        return tuple(values)
