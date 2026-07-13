"""vLLM backend for token-exact generation and synchronous weight updates."""

from __future__ import annotations

import asyncio
import itertools
import os
import socket
import uuid
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .models import InferenceRequest, InferenceResult


@dataclass(frozen=True, slots=True)
class VllmCheckpointWeightUpdate:
    """A complete local checkpoint to load before the next rollout version."""

    checkpoint_path: str | os.PathLike[str]

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_path, str) and not self.checkpoint_path:
            raise ValueError("checkpoint_path must not be empty")

        path = Path(self.checkpoint_path).expanduser().resolve()
        _validate_checkpoint_directory(path)
        object.__setattr__(self, "checkpoint_path", path)


@dataclass(frozen=True, slots=True)
class VllmNcclWeightUpdate:
    """Named trainer weights to send through vLLM's NCCL transfer backend."""

    named_weights: Iterable[tuple[str, Any]]

    def __post_init__(self) -> None:
        named_weights = tuple(self.named_weights)
        if not named_weights:
            raise ValueError("named_weights must not be empty")
        object.__setattr__(self, "named_weights", named_weights)


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
        weight_transfer: Literal["nccl", "checkpoint"] = "nccl",
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
                "pass weight_transfer directly instead of weight_transfer_config"
            )
        if weight_transfer not in {"nccl", "checkpoint"}:
            raise ValueError("weight_transfer must be 'nccl' or 'checkpoint'")
        if weight_transfer == "nccl":
            _validate_nccl_topology(options)
            options["weight_transfer_config"] = {"backend": "nccl"}

        try:
            from vllm import (
                AsyncEngineArgs,
                AsyncLLMEngine,
                SamplingParams,
                TokensPrompt,
            )
            if weight_transfer == "nccl":
                import torch
                from vllm.distributed.weight_transfer.base import (
                    WeightTransferInitRequest,
                    WeightTransferUpdateRequest,
                )
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLTrainerSendWeightsArgs,
                    NCCLWeightTransferEngine,
                )
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
        self._weight_transfer = weight_transfer
        self._inference_world_size = (
            int(options.get("data_parallel_size", 1))
            * int(options.get("pipeline_parallel_size", 1))
            * int(options.get("tensor_parallel_size", 1))
        )
        self._nccl_group: Any = None
        self._nccl_device: Any = None
        if weight_transfer == "nccl":
            self._torch = torch
            self._weight_transfer_init_request = WeightTransferInitRequest
            self._weight_transfer_update_request = WeightTransferUpdateRequest
            self._nccl_send_args = NCCLTrainerSendWeightsArgs
            self._nccl_engine = NCCLWeightTransferEngine

        self._request_prefix = f"chito-{uuid.uuid4().hex}"
        self._request_ids = itertools.count()
        self._lifecycle = asyncio.Condition()
        self._weight_update_lock = asyncio.Lock()
        self._active_requests = 0
        self._weight_update_in_progress = False
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
        """Synchronously replace vLLM weights before admitting new rollout."""
        self._validate_update_type(update)
        if not isinstance(new_policy_version, int) or isinstance(
            new_policy_version, bool
        ):
            raise TypeError("new_policy_version must be an integer")
        if new_policy_version < 0:
            raise ValueError("new_policy_version must be non-negative")

        async with self._weight_update_lock:
            if isinstance(update, VllmCheckpointWeightUpdate):
                _validate_checkpoint_directory(Path(update.checkpoint_path))
            else:
                self._validate_nccl_device(update)
            await self._begin_weight_update()
            failure: BaseException | None = None
            try:
                update_task = asyncio.create_task(self._apply_weight_update(update))
                try:
                    await asyncio.shield(update_task)
                except asyncio.CancelledError as cancellation:
                    try:
                        await update_task
                    except BaseException as update_error:
                        raise BaseExceptionGroup(
                            "weight update failed while it was cancelled",
                            [cancellation, update_error],
                        ) from cancellation
                    raise
            except BaseException as exc:
                failure = exc
                raise
            finally:
                await self._end_weight_update(failure)

    def _validate_update_type(self, update: object) -> None:
        expected = (
            VllmNcclWeightUpdate
            if self._weight_transfer == "nccl"
            else VllmCheckpointWeightUpdate
        )
        if not isinstance(update, expected):
            raise TypeError(
                f"{self._weight_transfer} transfer requires {expected.__name__}"
            )

    def _validate_nccl_device(self, update: VllmNcclWeightUpdate) -> None:
        device = update.named_weights[0][1].device
        if device.type != "cuda":
            raise ValueError("NCCL weights must be CUDA tensors")
        if any(weight.device != device for _, weight in update.named_weights):
            raise ValueError("all NCCL weights must be on the same CUDA device")
        if self._nccl_device is not None and device != self._nccl_device:
            raise ValueError("NCCL trainer device cannot change after initialization")

    async def _apply_weight_update(
        self, update: VllmNcclWeightUpdate | VllmCheckpointWeightUpdate
    ) -> None:
        await self._engine.pause_generation(mode="wait", clear_cache=True)
        if isinstance(update, VllmCheckpointWeightUpdate):
            await self._reload_checkpoint(update)
        else:
            await self._broadcast_nccl_weights(update)
        await self._engine.resume_generation()

    async def _reload_checkpoint(self, update: VllmCheckpointWeightUpdate) -> None:
        await self._engine.collective_rpc(
            "reload_weights",
            kwargs={
                "weights_path": str(update.checkpoint_path),
                "is_checkpoint_format": True,
            },
        )

    async def _broadcast_nccl_weights(self, update: VllmNcclWeightUpdate) -> None:
        device = update.named_weights[0][1].device
        await self._ensure_nccl_group(device)
        await self._engine.start_weight_update(is_checkpoint_format=True)

        update_request = self._weight_transfer_update_request(
            update_info={
                "names": [name for name, _ in update.named_weights],
                "dtype_names": [
                    str(weight.dtype).removeprefix("torch.")
                    for _, weight in update.named_weights
                ],
                "shapes": [
                    list(weight.shape) for _, weight in update.named_weights
                ],
                "packed": False,
            }
        )
        await self._wait_for_operations(
            self._engine.update_weights(update_request),
            asyncio.to_thread(self._send_nccl_weights, update),
        )
        await self._engine.finish_weight_update()

    async def _ensure_nccl_group(self, device: Any) -> None:
        if self._nccl_group is not None:
            return

        init_info = {
            "master_address": "127.0.0.1",
            "master_port": _find_free_local_port(),
            "rank_offset": 1,
            "world_size": self._inference_world_size + 1,
        }
        request = self._weight_transfer_init_request(init_info=init_info)
        receiver_init = self._engine.init_weight_transfer_engine(request)
        trainer_info = {
            "master_address": init_info["master_address"],
            "master_port": init_info["master_port"],
            "world_size": init_info["world_size"],
        }
        results = await asyncio.gather(
            receiver_init,
            asyncio.to_thread(self._init_nccl_trainer, device, trainer_info),
            return_exceptions=True,
        )
        group = results[1]
        try:
            self._raise_operation_errors(results)
        except BaseException:
            if not isinstance(group, BaseException):
                group.destroy()
            raise
        self._nccl_group = group
        self._nccl_device = device

    def _init_nccl_trainer(self, device: Any, init_info: dict[str, Any]) -> Any:
        with self._torch.cuda.device(device):
            return self._nccl_engine.trainer_init(init_info)

    def _send_nccl_weights(self, update: VllmNcclWeightUpdate) -> None:
        with self._torch.cuda.device(self._nccl_device):
            self._nccl_engine.trainer_send_weights(
                iter(update.named_weights),
                self._nccl_send_args(group=self._nccl_group, packed=False),
            )

    @staticmethod
    async def _wait_for_operations(
        first: Awaitable[Any], second: Awaitable[Any]
    ) -> tuple[Any, Any]:
        results = await asyncio.gather(first, second, return_exceptions=True)
        VllmBackend._raise_operation_errors(results)
        return results[0], results[1]

    @staticmethod
    def _raise_operation_errors(results: Sequence[Any]) -> None:
        errors = [result for result in results if isinstance(result, BaseException)]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("concurrent weight transfer failed", errors)

    async def aclose(self) -> None:
        """Wait for admitted operations and release vLLM resources once."""
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

        group = self._nccl_group
        try:
            self._engine.shutdown()
        finally:
            self._nccl_group = None
            self._nccl_device = None
            try:
                if group is not None:
                    group.destroy()
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
                self._weight_update_in_progress = False
                self._lifecycle.notify_all()
                raise

    async def _end_weight_update(self, failure: BaseException | None) -> None:
        async with self._lifecycle:
            if failure is not None and self._poisoned is None:
                self._poisoned = failure
            self._weight_update_in_progress = False
            self._lifecycle.notify_all()

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


def _validate_checkpoint_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"checkpoint path is not a directory: {path}")


def _validate_nccl_topology(options: Mapping[str, object]) -> None:
    if int(options.get("nnodes", 1)) != 1:
        raise ValueError("NCCL weight transfer currently requires one node")
    if int(options.get("prefill_context_parallel_size", 1)) != 1:
        raise ValueError(
            "NCCL weight transfer does not support prefill context parallelism"
        )
    if options.get("distributed_executor_backend") == "external_launcher":
        raise ValueError("NCCL weight transfer does not support external_launcher")
    if int(os.environ.get("VLLM_DP_SIZE", "1")) != 1:
        raise ValueError(
            "pass data_parallel_size through engine_kwargs instead of VLLM_DP_SIZE"
        )


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
