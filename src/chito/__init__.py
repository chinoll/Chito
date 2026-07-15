"""A small GRPO rollout service for distributed training."""

from .engine import RolloutEngine
from .models import (
    InferenceRequest,
    InferenceResult,
    RolloutConfig,
    RolloutGroup,
    RolloutPrompt,
    RolloutSample,
    TrainingBatch,
    TrainingSample,
)
from .protocols import InferenceBackend, RolloutContext, RolloutWorkflow
from .workflows import GRPOWorkflow, SingleTurnWorkflow

__version__ = "0.1.0"

__all__ = [
    "GRPOWorkflow",
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResult",
    "RolloutConfig",
    "RolloutContext",
    "RolloutEngine",
    "RolloutGroup",
    "RolloutPrompt",
    "RolloutSample",
    "RolloutWorkflow",
    "SingleTurnWorkflow",
    "TrainingBatch",
    "TrainingSample",
]
