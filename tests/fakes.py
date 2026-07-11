"""Deterministic backend used by the rollout engine tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from chito import InferenceRequest, InferenceResult


GenerateControl = Callable[[InferenceRequest], Awaitable[None]]
UpdateControl = Callable[[object, int], Awaitable[None]]


class FakeInferenceBackend:
    def __init__(
        self,
        *,
        generate_control: GenerateControl | None = None,
        update_control: UpdateControl | None = None,
    ) -> None:
        self.generate_control = generate_control
        self.update_control = update_control
        self.requests: list[InferenceRequest] = []
        self.updates: list[tuple[object, int, int]] = []
        self.inflight = 0
        self.closed_count = 0

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        self.inflight += 1
        try:
            if self.generate_control is not None:
                await self.generate_control(request)
            base = 1000 + request.sample_index * 10
            return InferenceResult(
                output_token_ids=(base, base + 1),
                output_logprobs=(-0.25, -0.5),
                policy_version=request.policy_version,
            )
        finally:
            self.inflight -= 1

    async def update_weights(
        self, update: object, *, new_policy_version: int
    ) -> None:
        self.updates.append((update, new_policy_version, self.inflight))
        if self.update_control is not None:
            await self.update_control(update, new_policy_version)

    async def aclose(self) -> None:
        self.closed_count += 1


async def wait_until(
    predicate: Callable[[], bool], *, attempts: int = 1000
) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")
