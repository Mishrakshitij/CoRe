"""
GRPO Loss (Group Relative Policy Optimization)
==============================================

Token-level policy optimization with group-normalized advantages.

Reference: DeepSeek-R1 (2024)

Loss formula:
    L_GRPO = -sum_j(A_j * log(pi_theta(tau_j|x))) + beta * KL(pi_theta || pi_ref)

Where:
    A_j = (R_j - mean(R)) / std(R)  # Group-normalized advantage
    R_j = reward for trace j
"""

import torch
from typing import Dict

from .base_loss import BasePolicyLoss, LossOutput


class GRPOLoss(BasePolicyLoss):
    """
    GRPO: Group Relative Policy Optimization

    Token-level importance sampling with hard clipping.
    """

    def __init__(self, config: dict):
        # Force token-level for standard GRPO
        config["importance_sampling_level"] = "token"
        super().__init__(config)

        # GRPO typically uses larger epsilon
        self.epsilon = config.get("epsilon", 0.2)
        self.epsilon_high = config.get("epsilon_high", 0.2)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> LossOutput:
        """
        Compute GRPO loss with token-level clipping.

        Args:
            log_probs: [batch, seq_len] current policy log probs
            old_log_probs: [batch, seq_len] old policy log probs
            ref_log_probs: [batch, seq_len] reference policy log probs
            advantages: [batch] group-normalized advantages
            mask: [batch, seq_len] attention mask
        """
        batch_size, seq_len = log_probs.shape

        # Ensure all tensors are on the same device (for multi-GPU setups)
        device = log_probs.device
        old_log_probs = old_log_probs.to(device)
        ref_log_probs = ref_log_probs.to(device)
        advantages = advantages.to(device)
        mask = mask.to(device)

        # Compute token-level importance ratio
        ratio = self.compute_importance_ratio(log_probs, old_log_probs, mask)

        # Expand advantages to token level if needed
        if advantages.dim() == 1:
            advantages_expanded = advantages.unsqueeze(-1).expand(-1, seq_len)
        else:
            advantages_expanded = advantages

        # Clipped ratio
        clipped_ratio = self.clip_ratio(ratio, advantages_expanded)

        # Policy loss: min(ratio * A, clipped_ratio * A)
        # For positive advantages: we want ratio to not be too high
        # For negative advantages: we want ratio to not be too low
        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded

        # Take minimum for positive advantages, maximum for negative
        policy_loss_per_token = -torch.where(
            advantages_expanded > 0,
            torch.min(surr1, surr2),
            torch.max(surr1, surr2),
        )

        # Mask and average
        policy_loss = (policy_loss_per_token * mask).sum() / mask.sum().clamp(min=1)

        # KL divergence from reference
        kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs, mask)

        # Total loss
        total_loss = policy_loss + self.beta * kl_loss

        # Metrics
        with torch.no_grad():
            clip_fraction = ((ratio - clipped_ratio).abs() > 1e-6).float()
            clip_fraction = (clip_fraction * mask).sum() / mask.sum().clamp(min=1)

            mask_sum = mask.sum().clamp(min=1).item()
            metrics = {
                "policy_loss": policy_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "mean_ratio": (ratio * mask).sum().item() / mask_sum,
                "clip_fraction": clip_fraction.item(),
                "mean_advantage": advantages.mean().item(),
            }

        return LossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )
