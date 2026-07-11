"""Backend, workflow, reward, and post-processing contracts."""

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
    """Backend boundary implemented by a vLLM or SGLang adapter."""

    async def generate(self, request: InferenceRequest) -> InferenceResult: ...

    async def update_weights(
        self, update: object, *, new_policy_version: int
    ) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RolloutContext:
    """Per-sample context supplied by the engine to a workflow."""

    backend: InferenceBackend
    sample_index: int
    policy_version: int


@runtime_checkable
class RolloutWorkflow(Protocol):
    """Produces one unrewarded sample; the engine owns grouping and rewards."""

    async def run(
        self, context: RolloutContext, prompt: RolloutPrompt
    ) -> RolloutSample: ...


@runtime_checkable
class RewardFunction(Protocol):
    """Asynchronously scores one completed sample."""

    async def __call__(
        self, prompt: RolloutPrompt, sample: RolloutSample
    ) -> float: ...


@runtime_checkable
class GroupPostHook(Protocol):
    """Accepts, replaces, or rejects one complete rewarded group."""

    async def __call__(self, group: RolloutGroup) -> RolloutGroup | None: ...
