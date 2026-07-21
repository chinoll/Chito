"""vLLM generation and the receiver half of NCCL weight updates."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import math
import os
import uuid
from collections.abc import Mapping
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
        initial_policy_version: int,
        engine_kwargs: Mapping[str, object],
        seed: int = 0,
    ) -> None:
        options = dict(engine_kwargs)
        reserved = {"logits_processors", "logprobs_mode", "weight_transfer_config"}
        duplicate = reserved.intersection(options)
        if duplicate:
            names = ", ".join(sorted(duplicate))
            raise ValueError(f"engine_kwargs must not override {names}")
        if not math.isfinite(float(temperature)) or temperature < 0.01:
            raise ValueError("temperature must be at least 0.01")
        if top_p != 1:
            raise ValueError("V1 requires top_p=1")
        if (
            not isinstance(initial_policy_version, int)
            or isinstance(initial_policy_version, bool)
            or initial_policy_version < 0
        ):
            raise ValueError("initial_policy_version must be a non-negative integer")

        options["logprobs_mode"] = "processed_logprobs"
        options["logits_processors"] = None
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
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            min_tokens=0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            logprobs=0,
            structured_outputs=None,
            logit_bias=None,
            allowed_token_ids=None,
            bad_words=None,
            extra_args=None,
            repetition_detection=None,
            detokenize=False,
        )
        self._seed = seed
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
        self._update_lock = asyncio.Lock()
        self._loaded_policy_version = initial_policy_version
        self._active_requests = 0
        self._updating = False
        self._channel_initialized = False
        self._receive_task: asyncio.Task[Any] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._poisoned: BaseException | None = None
        self._closing = False
        self._closed = False

    @property
    def inference_world_size(self) -> int:
        return self._inference_world_size

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate exact completion token IDs."""
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")

        loaded_policy_version = await self._begin_request(request.policy_version)
        try:
            prompt = self._tokens_prompt(
                prompt_token_ids=list(request.prompt.token_ids)
            )
            request_id = f"{self._request_prefix}-{next(self._request_ids)}"
            sampling_params = copy.copy(self._sampling_params)
            sampling_params.seed = self._request_seed(request)
            final_output = None
            async for output in self._engine.generate(
                prompt, sampling_params, request_id
            ):
                final_output = output
            return self._to_result(final_output, request, loaded_policy_version)
        finally:
            await self._finish_request()

    def _request_seed(self, request: InferenceRequest) -> int:
        key = (
            f"{self._seed}:{request.prompt.prompt_id}:{request.sample_index}"
        ).encode()
        return int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "little")

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
        async with self._update_lock:
            if not self._channel_initialized:
                raise RuntimeError("weight channel is not initialized")
            await self._begin_update()
            try:
                await self._engine.pause_generation(mode="wait", clear_cache=True)
                await self._check_update_open()
                await self._engine.start_weight_update()
                await self._check_update_open()
                request = self._update_request(update_info=dict(update_info))
                task = asyncio.create_task(self._engine.update_weights(request))
                self._receive_task = task
                return task
            except BaseException as exc:
                await self._end_update(None if self._closing else exc)
                raise

    async def finish_weight_update(
        self,
        receive_task: asyncio.Task[Any],
        new_policy_version: int,
    ) -> None:
        """Commit a completed receive and resume generation."""
        operation = asyncio.create_task(
            self._finish_weight_update(receive_task, new_policy_version)
        )
        await self._await_shielded(operation)

    async def _finish_weight_update(
        self,
        receive_task: asyncio.Task[Any],
        new_policy_version: int,
    ) -> None:
        async with self._update_lock:
            if receive_task is not self._receive_task:
                raise RuntimeError("receive task does not match the pending update")
            if (
                not isinstance(new_policy_version, int)
                or isinstance(new_policy_version, bool)
                or new_policy_version != self._loaded_policy_version + 1
            ):
                failure = ValueError(
                    "new_policy_version must be an integer that increments "
                    "the loaded version by one"
                )
                await self._await_receive(failure)
                await self._end_update(failure)
                raise failure
            try:
                await self._await_receive(None)
                await self._engine.finish_weight_update()
                await self._engine.resume_generation()
            except BaseException as exc:
                await self._end_update(None if self._closing else exc)
                raise
            await self._end_update(None, new_policy_version)

    async def fail_weight_update(self, message: str) -> None:
        """Poison the backend after the trainer sender failed."""
        operation = asyncio.create_task(self._fail_weight_update(message))
        await self._await_shielded(operation)

    async def _fail_weight_update(self, message: str) -> None:
        async with self._update_lock:
            failure = RuntimeError(message)
            await self._await_receive(failure)
            await self._end_update(failure)

    async def aclose(self) -> None:
        """Stop admitted work and release the vLLM engine."""
        async with self._condition:
            if self._close_task is None:
                self._closing = True
                self._condition.notify_all()
                self._close_task = asyncio.create_task(self._close())
            close_task = self._close_task

        await self._await_shielded(close_task)

    async def _close(self) -> None:
        receive_failure: BaseException | None = None
        shutdown_failure: BaseException | None = None

        async with self._update_lock:
            try:
                await self._await_receive(None)
            except BaseException as exc:
                receive_failure = exc

            async with self._condition:
                self._updating = False
                self._condition.notify_all()
                await self._condition.wait_for(lambda: self._active_requests == 0)

            try:
                self._engine.shutdown()
            except BaseException as exc:
                shutdown_failure = exc
            finally:
                async with self._condition:
                    self._closed = True
                    self._condition.notify_all()

        if receive_failure is not None:
            if shutdown_failure is not None:
                receive_failure.add_note(
                    "vLLM shutdown also failed: "
                    f"{type(shutdown_failure).__name__}: {shutdown_failure}"
                )
            raise receive_failure
        if shutdown_failure is not None:
            raise shutdown_failure

    async def _begin_request(self, requested_policy_version: int) -> int:
        async with self._condition:
            while self._updating:
                self._raise_if_unusable()
                await self._condition.wait()
            self._raise_if_unusable()
            if requested_policy_version != self._loaded_policy_version:
                raise ValueError(
                    "request policy_version does not match loaded vLLM weights"
                )
            loaded_policy_version = self._loaded_policy_version
            self._active_requests += 1
            return loaded_policy_version

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
            try:
                await self._condition.wait_for(
                    lambda: self._active_requests == 0 or self._closing or self._closed
                )
                self._raise_if_unusable()
            except BaseException:
                self._updating = False
                self._condition.notify_all()
                raise

    async def _check_update_open(self) -> None:
        async with self._condition:
            self._raise_if_unusable()

    async def _await_receive(self, primary: BaseException | None) -> None:
        task = self._receive_task
        if task is None:
            return
        try:
            await task
        except BaseException as cleanup:
            if primary is None:
                raise
            if cleanup is not primary:
                primary.add_note(
                    f"receiver failed: {type(cleanup).__name__}: {cleanup}"
                )
        finally:
            if task.done() and self._receive_task is task:
                self._receive_task = None

    @staticmethod
    async def _await_shielded(task: asyncio.Task[Any]) -> Any:
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    if cancellation is not None:
                        cancellation.add_note("the protected operation was cancelled")
                        raise cancellation
                    raise
                if cancellation is None:
                    cancellation = exc
                continue
            except BaseException as exc:
                if cancellation is None:
                    raise
                cancellation.add_note(
                    f"protected operation failed: {type(exc).__name__}: {exc}"
                )
                raise cancellation
            if cancellation is not None:
                raise cancellation
            return result

    async def _end_update(
        self,
        failure: BaseException | None,
        new_policy_version: int | None = None,
    ) -> None:
        async with self._condition:
            if failure is not None and self._poisoned is None:
                self._poisoned = failure
            if failure is None and new_policy_version is not None:
                self._loaded_policy_version = new_policy_version
            self._updating = False
            self._condition.notify_all()

    def _raise_if_unusable(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("vLLM backend is closed")
        if self._poisoned is not None:
            raise VllmBackendPoisonedError(self._poisoned) from self._poisoned

    @staticmethod
    def _to_result(
        output: Any,
        request: InferenceRequest,
        loaded_policy_version: int,
    ) -> InferenceResult:
        if output is None or not output.finished:
            raise RuntimeError("vLLM generation ended without a final output")
        if tuple(output.prompt_token_ids) != request.prompt.token_ids:
            raise RuntimeError("vLLM changed the supplied prompt token IDs")
        if len(output.outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected completion count")

        completion = output.outputs[0]
        token_ids = tuple(int(token_id) for token_id in completion.token_ids)
        output_logprobs = VllmBackend._chosen_logprobs(completion, token_ids)
        return InferenceResult(
            output_token_ids=token_ids,
            output_logprobs=output_logprobs,
            policy_version=loaded_policy_version,
            finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason,
        )

    @staticmethod
    def _chosen_logprobs(
        completion: Any, token_ids: tuple[int, ...]
    ) -> tuple[float, ...]:
        rows = completion.logprobs
        if rows is None or len(rows) != len(token_ids):
            raise RuntimeError("vLLM returned misaligned chosen-token logprobs")

        chosen = []
        for token_id, row in zip(token_ids, rows, strict=True):
            if row is None or token_id not in row:
                raise RuntimeError(
                    f"vLLM omitted chosen token {token_id} from logprobs"
                )
            chosen.append(float(row[token_id].logprob))
        return tuple(chosen)
