"""Public errors raised by the rollout engine."""

from __future__ import annotations

from .models import RolloutGroup


class RolloutError(RuntimeError):
    """Base class for rollout coordination failures."""


class EngineClosedError(RolloutError):
    """Raised when an operation requires an open engine."""


class RolloutAlreadyStartedError(RolloutError):
    """Raised when a second prompt source is registered."""


class RolloutNotStartedError(RolloutError):
    """Raised when a batch is requested before rollout starts."""


class RolloutFailedError(RolloutError):
    """Raised by consumers after the background producer fails."""

    def __init__(self, cause: BaseException):
        self.cause = cause
        super().__init__(f"rollout producer failed: {type(cause).__name__}: {cause}")


class SourceExhaustedError(RolloutError):
    """Raised when the source cannot provide another complete training batch."""

    def __init__(self, remaining_groups: tuple[RolloutGroup, ...]):
        self.remaining_groups = remaining_groups
        super().__init__(
            "prompt source exhausted with "
            f"{len(remaining_groups)} accepted group(s) remaining"
        )


class InvalidRolloutGroupError(RolloutError):
    """Raised when workflow or post-hook output breaks group invariants."""

