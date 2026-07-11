"""Immutable values exchanged by rollout components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field


def _as_int_tuple(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise TypeError(f"{name} must contain only integers")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must contain only non-negative integers")
    return result


@dataclass(frozen=True, slots=True)
class RolloutPrompt:
    """A tokenized prompt with a stable identity."""

    prompt_id: str
    token_ids: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty")
        object.__setattr__(
            self, "token_ids", _as_int_tuple(self.token_ids, "prompt.token_ids")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """One backend generation request."""

    prompt: RolloutPrompt
    sample_index: int
    policy_version: int

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Exact output tokens and behavior log-probabilities from a backend."""

    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    policy_version: int

    def __post_init__(self) -> None:
        tokens = _as_int_tuple(self.output_token_ids, "output_token_ids")
        logprobs = tuple(float(value) for value in self.output_logprobs)
        if not tokens:
            raise ValueError("inference output must contain at least one token")
        if len(tokens) != len(logprobs):
            raise ValueError("output_token_ids and output_logprobs must have equal length")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        object.__setattr__(self, "output_token_ids", tokens)
        object.__setattr__(self, "output_logprobs", logprobs)


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One training sample; reward is populated by the engine."""

    prompt_id: str
    sample_index: int
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    loss_mask: tuple[bool, ...]
    policy_version: int
    reward: float | None = None

    def __post_init__(self) -> None:
        tokens = _as_int_tuple(self.token_ids, "sample.token_ids")
        logprobs = tuple(float(value) for value in self.logprobs)
        loss_mask = tuple(bool(value) for value in self.loss_mask)
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if len(tokens) != len(logprobs) or len(tokens) != len(loss_mask):
            raise ValueError("token_ids, logprobs, and loss_mask must have equal length")
        if not any(loss_mask):
            raise ValueError("loss_mask must select at least one generated token")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "loss_mask", loss_mask)
        if self.reward is not None:
            object.__setattr__(self, "reward", float(self.reward))


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    """A complete GRPO group generated from one prompt and policy version."""

    prompt: RolloutPrompt
    samples: tuple[RolloutSample, ...]
    policy_version: int

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("rollout group must contain at least one sample")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        for sample in samples:
            if sample.reward is None:
                raise ValueError("every grouped sample must have a reward")
            if sample.prompt_id != self.prompt.prompt_id:
                raise ValueError("every grouped sample must match the group prompt_id")
            if sample.policy_version != self.policy_version:
                raise ValueError("all grouped samples must use the group policy_version")
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Exactly ``train_batch_size`` accepted rollout groups."""

    groups: tuple[RolloutGroup, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups:
            raise ValueError("training batch must contain at least one group")
        object.__setattr__(self, "groups", groups)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Fixed V1 concurrency and GRPO batching settings."""

    group_size: int
    train_batch_size: int
    max_concurrent_groups: int
    prefetch_batches: int = 1
    initial_policy_version: int = 0

    def __post_init__(self) -> None:
        positive = {
            "group_size": self.group_size,
            "train_batch_size": self.train_batch_size,
            "max_concurrent_groups": self.max_concurrent_groups,
            "prefetch_batches": self.prefetch_batches,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.initial_policy_version < 0:
            raise ValueError("initial_policy_version must be non-negative")

    @property
    def accepted_queue_capacity(self) -> int:
        return self.train_batch_size * self.prefetch_batches
