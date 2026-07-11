"""Built-in rollout workflows."""

from __future__ import annotations

from .models import InferenceRequest, RolloutPrompt, RolloutSample
from .protocols import RolloutContext


class SingleTurnWorkflow:
    """Runs one tokenized prompt through the configured inference backend."""

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
            raise ValueError(
                "backend returned a policy version different from the rollout context"
            )

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
