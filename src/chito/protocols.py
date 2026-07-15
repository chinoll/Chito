"""Small contracts used at the Workflow/backend boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import (
    InferenceRequest,
    InferenceResult,
    RolloutGroup,
    RolloutPrompt,
    RolloutSample,
)


@runtime_checkable
class InferenceBackend(Protocol):
    """Generation API exposed to a Workflow inside the rollout service."""

    async def generate(self, request: InferenceRequest) -> InferenceResult: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RolloutContext:
    """Per-completion state supplied by the service."""

    backend: InferenceBackend
    sample_index: int
    policy_version: int


@runtime_checkable
class RolloutWorkflow(Protocol):
    """Task and algorithm behavior owned by the rollout service."""

    async def setup(self) -> None: ...

    async def prepare(self, item: object, item_id: str) -> RolloutPrompt: ...

    async def run(
        self, context: RolloutContext, prompt: RolloutPrompt
    ) -> RolloutSample: ...

    async def reward(self, prompt: RolloutPrompt, sample: RolloutSample) -> float: ...

    async def postprocess(self, group: RolloutGroup) -> RolloutGroup | None: ...

    async def compute_advantages(self, group: RolloutGroup) -> tuple[float, ...]: ...

    async def aclose(self) -> None: ...
