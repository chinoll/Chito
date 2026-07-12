"""vLLM inference backend for token-exact rollout generation."""

from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .models import InferenceRequest, InferenceResult

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class VllmWeightUpdate:
    """One complete set of trainer tensors for a vLLM IPC update."""

    weights: Iterable[tuple[str, torch.Tensor]]
    checkpoint_format: bool = True
    packed: bool = False
    packed_buffer_size_bytes: int = 1 << 30

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

        self._request_prefix = f"chito-{uuid.uuid4().hex}"
        self._request_ids = itertools.count()
        self._lifecycle = asyncio.Condition()
        self._active_requests = 0
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
        """V1 does not claim unsupported in-process vLLM weight mutation."""
        raise NotImplementedError(
            "VllmBackend does not support dynamic weight updates in V1; "
            "close it and construct a new backend with the new checkpoint"
        )

    async def aclose(self) -> None:
        """Wait for admitted requests and release vLLM engine resources once."""
        async with self._lifecycle:
            if self._closed:
                return
            if self._closing:
                await self._lifecycle.wait_for(lambda: self._closed)
                return

            self._closing = True
            await self._lifecycle.wait_for(lambda: self._active_requests == 0)

        try:
            self._engine.shutdown()
        finally:
            async with self._lifecycle:
                self._closed = True
                self._lifecycle.notify_all()

    async def _begin_request(self) -> None:
        async with self._lifecycle:
            if self._closing or self._closed:
                raise RuntimeError("VllmBackend is closed")
            self._active_requests += 1

    async def _finish_request(self) -> None:
        async with self._lifecycle:
            self._active_requests -= 1
            self._lifecycle.notify_all()

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
