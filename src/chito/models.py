"""Public values exchanged by rollout and training processes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


def _int_tuple(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise TypeError(f"{name} must contain only integers")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must contain only non-negative integers")
    return result


def _finite_float_tuple(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class RolloutPrompt:
    """A tokenized prompt prepared from one dataset item."""

    prompt_id: str
    token_ids: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty")
        tokens = _int_tuple(self.token_ids, "token_ids")
        if not tokens:
            raise ValueError("token_ids must not be empty")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """One generation request sent from a Workflow to the backend."""

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
    """Exact completion tokens returned by the inference backend."""

    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    policy_version: int
    finish_reason: str
    stop_reason: int | str | None = None

    def __post_init__(self) -> None:
        tokens = _int_tuple(self.output_token_ids, "output_token_ids")
        logprobs = _finite_float_tuple(self.output_logprobs, "output_logprobs")
        if not tokens:
            raise ValueError("inference output must contain at least one token")
        if len(tokens) != len(logprobs):
            raise ValueError(
                "output_token_ids and output_logprobs must have equal length"
            )
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise ValueError("finish_reason must be a non-empty string")
        if isinstance(self.stop_reason, bool) or not isinstance(
            self.stop_reason, (int, str, type(None))
        ):
            raise TypeError("stop_reason must be an integer, string, or None")
        object.__setattr__(self, "output_token_ids", tokens)
        object.__setattr__(self, "output_logprobs", logprobs)


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One completion during rollout; reward is filled by the service."""

    prompt_id: str
    sample_index: int
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    policy_version: int
    finish_reason: str
    stop_reason: int | str | None = None
    reward: float | None = None

    def __post_init__(self) -> None:
        tokens = _int_tuple(self.token_ids, "token_ids")
        loss_mask = tuple(bool(value) for value in self.loss_mask)
        behavior_logprobs = _finite_float_tuple(
            self.behavior_logprobs, "behavior_logprobs"
        )
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if len(tokens) != len(loss_mask) or len(tokens) != len(behavior_logprobs):
            raise ValueError(
                "token_ids, loss_mask, and behavior_logprobs must have equal length"
            )
        if not any(loss_mask):
            raise ValueError("loss_mask must select at least one generated token")
        if loss_mask[0]:
            raise ValueError("loss_mask cannot select the first token")
        if any(
            logprob != 0.0
            for logprob, selected in zip(behavior_logprobs, loss_mask, strict=True)
            if not selected
        ):
            raise ValueError("masked tokens must have zero behavior_logprobs")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise ValueError("finish_reason must be a non-empty string")
        if isinstance(self.stop_reason, bool) or not isinstance(
            self.stop_reason, (int, str, type(None))
        ):
            raise TypeError("stop_reason must be an integer, string, or None")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "loss_mask", loss_mask)
        object.__setattr__(self, "behavior_logprobs", behavior_logprobs)
        if self.reward is not None:
            object.__setattr__(self, "reward", float(self.reward))


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    """A complete, rewarded GRPO group from one prompt."""

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
                raise ValueError("grouped samples must match the prompt_id")
            if sample.policy_version != self.policy_version:
                raise ValueError("grouped samples must use the group policy_version")
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One rollout sample after its complete group advantage is computed."""

    prompt_id: str
    sample_index: int
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    reward: float
    advantage: float
    policy_version: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens = _int_tuple(self.token_ids, "token_ids")
        loss_mask = tuple(bool(value) for value in self.loss_mask)
        behavior_logprobs = _finite_float_tuple(
            self.behavior_logprobs, "behavior_logprobs"
        )
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if len(tokens) != len(loss_mask) or len(tokens) != len(behavior_logprobs):
            raise ValueError(
                "token_ids, loss_mask, and behavior_logprobs must have equal length"
            )
        if not any(loss_mask):
            raise ValueError("loss_mask must select at least one generated token")
        if loss_mask[0]:
            raise ValueError("loss_mask cannot select the first token")
        if any(
            logprob != 0.0
            for logprob, selected in zip(behavior_logprobs, loss_mask, strict=True)
            if not selected
        ):
            raise ValueError("masked tokens must have zero behavior_logprobs")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if not math.isfinite(float(self.advantage)):
            raise ValueError("advantage must be finite")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "loss_mask", loss_mask)
        object.__setattr__(self, "behavior_logprobs", behavior_logprobs)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "advantage", float(self.advantage))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """The fixed contiguous sample shard returned to one training rank."""

    batch_id: int
    policy_version: int
    samples: tuple[TrainingSample, ...]
    global_sample_count: int

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if self.batch_id < 0:
            raise ValueError("batch_id must be non-negative")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not samples:
            raise ValueError("training batch must contain at least one sample")
        if self.global_sample_count < len(samples):
            raise ValueError(
                "global_sample_count cannot be smaller than the local shard"
            )
        if any(sample.policy_version != self.policy_version for sample in samples):
            raise ValueError("all samples must match the batch policy_version")
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Flat V1 configuration for one GRPO rollout service."""

    model: str
    group_size: int
    per_device_train_batch_size: int
    max_concurrent_groups: int
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    backend: Literal["vllm"] = "vllm"
    weight_transfer: Literal["nccl"] = "nccl"
    rollout_mode: Literal["sync", "async"] = "sync"
    rollout_gpu_ids: tuple[int, ...] = ()
    shuffle: bool = True
    seed: int = 0
    initial_policy_version: int = 0
    backend_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        positive = {
            "group_size": self.group_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "max_concurrent_groups": self.max_concurrent_groups,
            "max_tokens": self.max_tokens,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if not math.isfinite(float(self.temperature)) or self.temperature < 0.01:
            raise ValueError("temperature must be at least 0.01")
        if self.top_p != 1:
            raise ValueError("V1 requires top_p=1")
        if self.backend != "vllm":
            raise ValueError("V1 only supports backend='vllm'")
        if self.weight_transfer != "nccl":
            raise ValueError("V1 only supports weight_transfer='nccl'")
        if self.rollout_mode not in ("sync", "async"):
            raise ValueError("rollout_mode must be 'sync' or 'async'")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.initial_policy_version < 0:
            raise ValueError("initial_policy_version must be non-negative")

        gpu_ids = _int_tuple(tuple(self.rollout_gpu_ids), "rollout_gpu_ids")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("rollout_gpu_ids must not contain duplicates")
        kwargs = dict(self.backend_kwargs)
        reserved = {
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "device_ids",
            "logits_processors",
            "logprobs_mode",
            "weight_transfer_config",
        }
        duplicate = reserved.intersection(kwargs)
        if duplicate:
            names = ", ".join(sorted(duplicate))
            raise ValueError(f"backend_kwargs must not override {names}")
        object.__setattr__(self, "rollout_gpu_ids", gpu_ids)
        object.__setattr__(self, "backend_kwargs", kwargs)

    def global_sample_count(self, train_world_size: int) -> int:
        if train_world_size <= 0:
            raise ValueError("train_world_size must be positive")
        return self.per_device_train_batch_size * train_world_size
