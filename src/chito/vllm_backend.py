"""vLLM inference backend for token-exact rollout generation."""

from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .models import InferenceRequest, InferenceResult

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class VllmWeightUpdate:
    """One complete set of trainer tensors for a vLLM IPC update."""

    weights: Iterable[tuple[str, torch.Tensor]]
    checkpoint_format: bool = True
    packed: bool = True
    packed_buffer_size_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        weights = tuple(self.weights)
        if not weights:
            raise ValueError("weights must contain at least one named tensor")

        names: set[str] = set()
        for item in weights:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("weights must contain (name, tensor) tuples")
            name, _tensor = item
            if not isinstance(name, str) or not name:
                raise ValueError("every weight name must be a non-empty string")
            if name in names:
                raise ValueError(f"duplicate weight name: {name}")
            names.add(name)

        if not isinstance(self.checkpoint_format, bool):
            raise TypeError("checkpoint_format must be a boolean")
        if not isinstance(self.packed, bool):
            raise TypeError("packed must be a boolean")
        size = self.packed_buffer_size_bytes
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("packed_buffer_size_bytes must be an integer")
        if size <= 0:
            raise ValueError("packed_buffer_size_bytes must be positive")

        object.__setattr__(self, "weights", weights)


class VllmBackendPoisonedError(RuntimeError):
    """The loaded engine may contain a partial update and must be reloaded."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(
            "VllmBackend is poisoned by a failed weight update; "
            "close it and load a fresh backend"
        )
        self.cause = cause


class VllmBackend:
    """Load one model with vLLM's asynchronous Python engine.

    vLLM is imported only while constructing this backend, so the rest of
    :mod:`chito` remains usable without the optional dependency installed.
    """

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        engine_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError("max_tokens must be an integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")

        options = dict(engine_kwargs or {})
        if "model" in options:
            raise ValueError("pass model directly instead of through engine_kwargs")
        if "weight_transfer_config" in options:
            raise ValueError(
                "VllmBackend manages weight_transfer_config for its IPC backend"
            )
        options["weight_transfer_config"] = {"backend": "ipc"}

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
            from vllm.distributed.weight_transfer.ipc_engine import (
                IPCTrainerSendWeightsArgs,
                IPCWeightTransferEngine,
            )
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "vllm":
                raise
            raise ModuleNotFoundError(
                "VllmBackend requires the optional 'vllm' dependency; "
                "install chito[vllm]"
            ) from exc

        engine_args = AsyncEngineArgs(model=model, **options)
        self._engine: Any = AsyncLLMEngine.from_engine_args(engine_args)
        self._sampling_params: Any = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            logprobs=0,
            detokenize=False,
        )
        self._tokens_prompt = TokensPrompt
        self._weight_transfer_init_request = WeightTransferInitRequest
        self._weight_transfer_update_request = WeightTransferUpdateRequest
        self._ipc_sender_args = IPCTrainerSendWeightsArgs
        self._ipc_send_weights = IPCWeightTransferEngine.trainer_send_weights
        self._torch = torch

        self._request_prefix = f"chito-{uuid.uuid4().hex}"
        self._request_ids = itertools.count()
        self._lifecycle = asyncio.Condition()
        self._weight_update_lock = asyncio.Lock()
        self._active_requests = 0
        self._weight_update_in_progress = False
        self._weight_transfer_initialized = False
        self._poisoned: BaseException | None = None
        self._closing = False
        self._closed = False

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Generate from the request's exact prompt IDs and preserve logprobs."""
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")

        await self._begin_request()
        try:
            request_id = f"{self._request_prefix}-{next(self._request_ids)}"
            prompt = self._tokens_prompt(
                prompt_token_ids=list(request.prompt.token_ids)
            )
            final_output = None
            async for output in self._engine.generate(
                prompt,
                self._sampling_params,
                request_id,
            ):
                final_output = output
            return self._to_result(final_output, request)
        finally:
            await self._finish_request()

    async def update_weights(
        self, update: object, *, new_policy_version: int
    ) -> None:
        """Transfer one complete policy through vLLM's same-host IPC backend."""
        if not isinstance(update, VllmWeightUpdate):
            raise TypeError("update must be a VllmWeightUpdate")
        if not isinstance(new_policy_version, int) or isinstance(
            new_policy_version, bool
        ):
            raise TypeError("new_policy_version must be an integer")
        if new_policy_version < 0:
            raise ValueError("new_policy_version must be non-negative")

        async with self._weight_update_lock:
            tensor_device = self._validate_weight_tensors(update)
            await self._begin_weight_update()
            poison_cause: BaseException | None = None
            poison_required = True
            try:
                await self._engine.pause_generation(mode="wait", clear_cache=True)
                try:
                    await self._ensure_weight_transfer_initialized()
                except BaseException as init_error:
                    try:
                        await self._engine.resume_generation()
                    except BaseException as resume_error:
                        poison_required = True
                        raise BaseExceptionGroup(
                            "weight-transfer initialization and resume failed",
                            [init_error, resume_error],
                        ) from init_error
                    poison_required = False
                    raise

                await self._engine.start_weight_update(
                    is_checkpoint_format=update.checkpoint_format
                )
                try:
                    await self._send_weight_tensors(update, tensor_device)
                except BaseException as transfer_error:
                    try:
                        await self._engine.finish_weight_update()
                    except BaseException as finish_error:
                        raise BaseExceptionGroup(
                            "weight transfer and finish both failed",
                            [transfer_error, finish_error],
                        ) from transfer_error
                    raise

                await self._engine.finish_weight_update()
                await self._engine.resume_generation()
                poison_required = False
            except BaseException as exc:
                if poison_required:
                    poison_cause = exc
                raise
            finally:
                await self._end_weight_update(poison_cause)

    async def aclose(self) -> None:
        """Wait for admitted requests and release vLLM engine resources once."""
        async with self._lifecycle:
            if self._closed:
                return
            if self._closing:
                await self._lifecycle.wait_for(lambda: self._closed)
                return

            self._closing = True
            self._lifecycle.notify_all()
            await self._lifecycle.wait_for(
                lambda: self._active_requests == 0
                and not self._weight_update_in_progress
            )

        try:
            self._engine.shutdown()
        finally:
            async with self._lifecycle:
                self._closed = True
                self._lifecycle.notify_all()

    async def _begin_request(self) -> None:
        async with self._lifecycle:
            while self._weight_update_in_progress:
                self._raise_if_unusable()
                await self._lifecycle.wait()
            self._raise_if_unusable()
            self._active_requests += 1

    async def _finish_request(self) -> None:
        async with self._lifecycle:
            self._active_requests -= 1
            self._lifecycle.notify_all()

    async def _begin_weight_update(self) -> None:
        async with self._lifecycle:
            self._raise_if_unusable()
            self._weight_update_in_progress = True
            self._lifecycle.notify_all()
            try:
                while self._active_requests > 0:
                    if self._closing:
                        raise RuntimeError("VllmBackend is closed")
                    await self._lifecycle.wait()
            except BaseException:
                if self._weight_update_in_progress:
                    self._weight_update_in_progress = False
                    self._lifecycle.notify_all()
                raise

    async def _end_weight_update(self, poison_cause: BaseException | None) -> None:
        async with self._lifecycle:
            if poison_cause is not None and self._poisoned is None:
                self._poisoned = poison_cause
            self._weight_update_in_progress = False
            self._lifecycle.notify_all()

    async def _ensure_weight_transfer_initialized(self) -> None:
        if self._weight_transfer_initialized:
            return
        request = self._weight_transfer_init_request(init_info={})
        await self._engine.init_weight_transfer_engine(request)
        self._weight_transfer_initialized = True

    async def _send_weight_tensors(
        self, update: VllmWeightUpdate, tensor_device: int
    ) -> None:
        loop = asyncio.get_running_loop()

        def send_to_engine(update_info: Any) -> None:
            request = self._weight_transfer_update_request(
                update_info=asdict(update_info)
            )
            receiver = asyncio.run_coroutine_threadsafe(
                self._engine.update_weights(request), loop
            )
            receiver.result()

        sender_args = self._ipc_sender_args(
            send_mode=send_to_engine,
            packed=update.packed,
            packed_buffer_size_bytes=update.packed_buffer_size_bytes,
        )

        def send_from_tensor_device() -> None:
            with self._torch.cuda.device(tensor_device):
                self._ipc_send_weights(iter(update.weights), sender_args)

        sender = asyncio.create_task(asyncio.to_thread(send_from_tensor_device))
        try:
            await asyncio.shield(sender)
        except asyncio.CancelledError as cancellation:
            try:
                await sender
            except BaseException as sender_error:
                raise BaseExceptionGroup(
                    "weight sender failed while update was cancelled",
                    [cancellation, sender_error],
                ) from cancellation
            raise

    def _validate_weight_tensors(self, update: VllmWeightUpdate) -> int:
        tensor_device = None
        tensor_device_index: int | None = None
        for name, tensor in update.weights:
            if not isinstance(tensor, self._torch.Tensor):
                raise TypeError(f"weight {name!r} must be a torch.Tensor")
            if not tensor.is_cuda:
                raise ValueError(f"weight {name!r} must be on a CUDA device")
            if tensor_device is None:
                tensor_device = tensor.device
                tensor_device_index = tensor.device.index
            elif tensor.device != tensor_device:
                raise ValueError("all weights must be on the same CUDA device")

        if tensor_device_index is None:
            raise ValueError("weight CUDA device must have an explicit index")
        return tensor_device_index

    def _raise_if_unusable(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("VllmBackend is closed")
        if self._poisoned is not None:
            raise VllmBackendPoisonedError(self._poisoned) from self._poisoned

    @staticmethod
    def _to_result(output: Any, request: InferenceRequest) -> InferenceResult:
        if output is None or not output.finished:
            raise RuntimeError("vLLM generation ended without a final output")
        if tuple(output.prompt_token_ids) != request.prompt.token_ids:
            raise RuntimeError("vLLM did not preserve the supplied prompt token IDs")
        if len(output.outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected number of completions")

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
                raise RuntimeError(
                    "vLLM omitted a sampled token from its logprob output"
                )
            values.append(float(sampled.logprob))
        return tuple(values)
