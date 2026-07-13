"""Asynchronous rollout coordination for reinforcement learning."""

from .models import (
    InferenceRequest,
    InferenceResult,
    RolloutConfig,
    RolloutGroup,
    RolloutPrompt,
    RolloutSample,
    TrainingBatch,
)
from .engine import RolloutEngine
from .errors import (
    EngineClosedError,
    InvalidRolloutGroupError,
    RolloutAlreadyStartedError,
    RolloutError,
    RolloutFailedError,
    RolloutNotStartedError,
    SourceExhaustedError,
)
from .protocols import (
    GroupPostHook,
    InferenceBackend,
    RewardFunction,
    RolloutContext,
    RolloutWorkflow,
)
from .workflows import SingleTurnWorkflow
from .vllm_backend import (
    VllmBackend,
    VllmBackendPoisonedError,
    VllmCheckpointWeightUpdate,
)

__version__ = "0.1.0"

__all__ = [
    "GroupPostHook",
    "EngineClosedError",
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResult",
    "InvalidRolloutGroupError",
    "RewardFunction",
    "RolloutConfig",
    "RolloutContext",
    "RolloutAlreadyStartedError",
    "RolloutEngine",
    "RolloutError",
    "RolloutFailedError",
    "RolloutGroup",
    "RolloutPrompt",
    "RolloutNotStartedError",
    "RolloutSample",
    "RolloutWorkflow",
    "SingleTurnWorkflow",
    "SourceExhaustedError",
    "TrainingBatch",
    "VllmBackend",
    "VllmBackendPoisonedError",
    "VllmCheckpointWeightUpdate",
]
