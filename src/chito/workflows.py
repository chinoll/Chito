"""Default single-turn GRPO Workflow."""

from __future__ import annotations

import math

from .models import InferenceRequest, RolloutGroup, RolloutPrompt, RolloutSample
from .protocols import RolloutContext


class GRPOWorkflow:
    """Single-turn GRPO defaults; tasks usually implement prepare and reward."""

    async def setup(self) -> None:
        """Load tokenizer or task resources inside the rollout process."""

    async def prepare(self, item: object, item_id: str) -> RolloutPrompt:
        raise NotImplementedError

    async def run(
        self, context: RolloutContext, prompt: RolloutPrompt
    ) -> RolloutSample:
        result = await context.backend.generate(
            InferenceRequest(
                prompt=prompt,
                sample_index=context.sample_index,
                policy_version=context.policy_version,
            )
        )
        if result.policy_version != context.policy_version:
            raise ValueError("backend returned a different policy_version")

        prompt_length = len(prompt.token_ids)
        output_length = len(result.output_token_ids)
        return RolloutSample(
            prompt_id=prompt.prompt_id,
            sample_index=context.sample_index,
            token_ids=prompt.token_ids + result.output_token_ids,
            logprobs=(0.0,) * prompt_length + result.output_logprobs,
            loss_mask=(False,) * prompt_length + (True,) * output_length,
            policy_version=context.policy_version,
        )

    async def reward(self, prompt: RolloutPrompt, sample: RolloutSample) -> float:
        raise NotImplementedError

    async def postprocess(self, group: RolloutGroup) -> RolloutGroup | None:
        return group

    async def compute_advantages(self, group: RolloutGroup) -> tuple[float, ...]:
        rewards = tuple(float(sample.reward) for sample in group.samples)
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        std = math.sqrt(variance)
        return tuple((reward - mean) / (std + 1e-4) for reward in rewards)

    async def aclose(self) -> None:
        """Release resources created by setup."""


# The V1 default is a single model turn. Keep the old descriptive name as an
# alias without adding another wrapper class.
SingleTurnWorkflow = GRPOWorkflow
