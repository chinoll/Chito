"""Pure PyTorch Geometric-Mean Policy Optimization loss."""

from __future__ import annotations

import torch


def gmpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor | None,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    epsilon_low: float = 0.4,
    epsilon_high: float | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute GMPO from selected-token log-probabilities.

    GMPO clips each token log-ratio on the side selected by the advantage,
    averages the clipped log-ratios over each completion, exponentiates that
    average, and finally averages the sequence losses.

    When ``old_per_token_logps`` is ``None``, the current log-probabilities are
    detached and used as the old policy. The ratio is then one in value while
    retaining the policy-gradient derivative. Both epsilon values are bounds
    in log space; ``epsilon_high`` defaults to ``epsilon_low``. A missing
    ``completion_mask`` selects every token.
    """
    if per_token_logps.ndim != 2:
        raise ValueError("per_token_logps must have shape (batch, tokens)")
    if completion_mask is not None and completion_mask.shape != per_token_logps.shape:
        raise ValueError("completion_mask must match per_token_logps")
    if advantages.shape != per_token_logps.shape[:1]:
        raise ValueError("advantages must have shape (batch,)")
    if (
        old_per_token_logps is not None
        and old_per_token_logps.shape != per_token_logps.shape
    ):
        raise ValueError("old_per_token_logps must match per_token_logps")

    epsilon_high = epsilon_low if epsilon_high is None else epsilon_high
    if epsilon_low < 0 or epsilon_high < 0:
        raise ValueError("GMPO clipping values must be non-negative")

    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()

    mask = (
        torch.ones_like(per_token_logps)
        if completion_mask is None
        else completion_mask.to(dtype=per_token_logps.dtype)
    )
    log_ratio = per_token_logps - old_per_token_logps
    clamped_log_ratio = torch.clamp(
        log_ratio,
        min=-epsilon_low,
        max=epsilon_high,
    )

    advantages_by_token = advantages.unsqueeze(1)
    clipped_log_ratio = torch.where(
        advantages_by_token > 0,
        torch.minimum(log_ratio, clamped_log_ratio),
        torch.maximum(log_ratio, clamped_log_ratio),
    )

    completion_lengths = mask.sum(-1).clamp(min=1.0)
    sequence_log_ratio = (clipped_log_ratio * mask).sum(-1) / completion_lengths
    sequence_ratio = torch.exp(sequence_log_ratio)
    loss = (-sequence_ratio * advantages).mean()

    token_count = mask.sum().clamp(min=1.0)
    high_clipped = (log_ratio > epsilon_high) & (advantages_by_token > 0)
    low_clipped = (log_ratio < -epsilon_low) & (advantages_by_token < 0)
    low_clip = ((low_clipped * mask).sum() / token_count).detach()
    high_clip = ((high_clipped * mask).sum() / token_count).detach()
    clip_ratio = (((low_clipped | high_clipped) * mask).sum() / token_count).detach()
    return loss, (low_clip, high_clip, clip_ratio)
