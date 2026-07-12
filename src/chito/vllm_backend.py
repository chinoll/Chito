"""vLLM backend for token-exact generation and checkpoint reloads."""

from __future__ import annotations

import asyncio
import itertools
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import InferenceRequest, InferenceResult


@dataclass(frozen=True, slots=True)
class VllmWeightUpdate:
    """A complete local checkpoint to load before the next rollout version."""

    checkpoint_path: str | os.PathLike[str]

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_path, str) and not self.checkpoint_path:
            raise ValueError("checkpoint_path must not be empty")

        path = Path(self.checkpoint_path).expanduser().resolve()
        _validate_checkpoint_directory(path)
        object.__setattr__(self, "checkpoint_path", path)


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
        """Synchronously reload one complete local checkpoint into vLLM."""
        if not isinstance(update, VllmWeightUpdate):
            raise TypeError("update must be a VllmWeightUpdate")
        if not isinstance(new_policy_version, int) or isinstance(
            new_policy_version, bool
        ):
            raise TypeError("new_policy_version must be an integer")
        if new_policy_version < 0:
            raise ValueError("new_policy_version must be non-negative")

        async with self._weight_update_lock:
            checkpoint_path = Path(update.checkpoint_path)
            _validate_checkpoint_directory(checkpoint_path)
            await self._begin_weight_update()
            failure: BaseException | None = None
            try:
                await self._engine.pause_generation(mode="wait", clear_cache=True)
                await self._engine.collective_rpc(
                    "reload_weights",
                    kwargs={
                        "weights_path": str(checkpoint_path),
                        "is_checkpoint_format": True,
                    },
                )
                await self._engine.resume_generation()
            except BaseException as exc:
                failure = exc
                raise
            finally:
                await self._end_weight_update(failure)

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
