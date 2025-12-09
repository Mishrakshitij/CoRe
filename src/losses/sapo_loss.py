"""
SAPO Loss (Soft Adaptive Policy Optimization)
==============================================

Soft gating policy optimization with temperature-controlled sigmoid.

Reference: arxiv.org/abs/2511.20347 (Qwen Team)

Key features:
1. Replaces hard clipping with smooth sigmoid gate
2. Asymmetric temperatures for positive/negative advantages
3. Token-adaptive while maintaining sequence coherence
4. Better sample efficiency than GSPO

Loss formula:
    L_SAPO = -sum_t(g_t * A_t * log(pi_theta))

Where:
    g_t = sigma(tau * (r_t - 1)) * (4 / tau)
    tau = tau_pos if A > 0 else tau_neg
    r_t = pi_theta / pi_old (importance ratio)
"""

import torch
import torch.nn.functional as F
from typing import Dict

from .base_loss import BasePolicyLoss, LossOutput


class SAPOLoss(BasePolicyLoss):
    """
    SAPO: Soft Adaptive Policy Optimization

    Temperature-controlled soft gating for stable policy updates.
    """

    def __init__(self, config: dict):
        # SAPO uses token-level by default
        config["importance_sampling_level"] = "token"
        super().__init__(config)

        # SAPO-specific temperatures
        self.tau_pos = config.get("tau_pos", 1.0)
        self.tau_neg = config.get("tau_neg", 1.05)  # Faster decay for negative

        # Special temperature for rescue traces
        self.tau_rescue = config.get("tau_rescue", 0.8)

        # KL coefficient
        self.beta = config.get("beta", 0.04)

    def soft_gate(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        is_rescue: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute soft sigmoid gate based on importance ratio.

        g_t = sigma(tau * (r_t - 1)) * (4 / tau)

        This creates a smooth trust region that:
        - Peaks at r_t = 1 (on-policy)
        - Decays smoothly as ratio deviates
        - Asymmetric decay for positive/negative advantages
        """
        # Select temperature based on advantage sign
        if is_rescue is not None and is_rescue.any():
            # Special handling for rescue traces (wider gate)
            tau = torch.where(
                is_rescue.unsqueeze(-1).expand_as(advantages) if is_rescue.dim() == 1 else is_rescue,
                torch.full_like(advantages, self.tau_rescue),
                torch.where(
                    advantages > 0,
                    torch.full_like(advantages, self.tau_pos),
                    torch.full_like(advantages, self.tau_neg),
                ),
            )
        else:
            tau = torch.where(
                advantages > 0,
                torch.full_like(advantages, self.tau_pos),
                torch.full_like(advantages, self.tau_neg),
            )

        # Soft gate: sigmoid centered at r=1
        gate = torch.sigmoid(tau * (ratio - 1.0))

        # Scaling factor to normalize gradient magnitude
        scaling = 4.0 / tau

        return gate * scaling

    def compute_weight_function(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weight function for gradient.

        w_t = 4 * p_t * (1 - p_t)
        where p_t = sigma(tau * (r_t - 1))

        This weight peaks at r=1 and decays smoothly.
        """
        tau = torch.where(
            advantages > 0,
            torch.full_like(advantages, self.tau_pos),
            torch.full_like(advantages, self.tau_neg),
        )

        p = torch.sigmoid(tau * (ratio - 1.0))
        weight = 4.0 * p * (1.0 - p)

        return weight

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        is_rescue: torch.Tensor = None,
    ) -> LossOutput:
        """
        Compute SAPO loss with soft gating.

        Args:
            log_probs: [batch, seq_len] current policy log probs
            old_log_probs: [batch, seq_len] old policy log probs
            ref_log_probs: [batch, seq_len] reference policy log probs
            advantages: [batch] or [batch, seq_len] advantages
            mask: [batch, seq_len] attention mask
            is_rescue: [batch] optional flag for rescue traces
        """
        batch_size, seq_len = log_probs.shape

        # Compute token-level importance ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Expand advantages to token level if needed
        if advantages.dim() == 1:
            advantages_expanded = advantages.unsqueeze(-1).expand(-1, seq_len)
        else:
            advantages_expanded = advantages

        # Compute soft gate
        gate = self.soft_gate(ratio, advantages_expanded, is_rescue)

        # SAPO loss: -g_t * A_t * log(pi)
        # Note: We use the gated advantage directly, not clipped ratio
        policy_loss_per_token = -gate * advantages_expanded * log_probs

        # Mask and average
        policy_loss = (policy_loss_per_token * mask).sum() / mask.sum().clamp(min=1)

        # Alternative formulation using weight function (for gradient analysis)
        # weight = self.compute_weight_function(ratio, advantages_expanded)
        # weighted_gradient = weight * ratio * advantages_expanded

        # KL divergence from reference
        kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs, mask)

        # Total loss
        total_loss = policy_loss + self.beta * kl_loss

        # Metrics
        with torch.no_grad():
            # Effective clip fraction (tokens where gate < 0.5)
            low_gate_fraction = ((gate < 0.5).float() * mask).sum() / mask.sum().clamp(min=1)

            # Gate statistics
            masked_gate = gate * mask
            gate_sum = masked_gate.sum()
            gate_mean = gate_sum / mask.sum().clamp(min=1)

            metrics = {
                "policy_loss": policy_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "mean_ratio": (ratio * mask).sum().item() / mask.sum().item(),
                "mean_gate": gate_mean.item(),
                "low_gate_fraction": low_gate_fraction.item(),
                "mean_advantage": advantages.mean().item(),
                "tau_pos": self.tau_pos,
                "tau_neg": self.tau_neg,
            }

        return LossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        is_rescue: torch.Tensor = None,
    ) -> LossOutput:
        """Forward with optional rescue flag."""
        return self.compute_loss(
            log_probs, old_log_probs, ref_log_probs,
            advantages, mask, is_rescue
        )
