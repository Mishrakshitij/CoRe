"""
GSPO Loss (Group Sequence Policy Optimization)
==============================================

Sequence-level policy optimization for better stability and efficiency.

Reference: arxiv.org/abs/2507.18071 (Qwen Team)

Key differences from GRPO:
1. Importance ratio computed at sequence level (geometric mean)
2. Sequence-level clipping
3. Better stability for MoE models
4. No need for Routing Replay

Loss formula:
    L_GSPO = -sum_i(w_i * A_i * log(pi_theta(y_i|x)))

Where:
    w_i = exp(1/|y_i| * sum_t(log(pi_theta/pi_old)))  # Sequence importance
    Clipped to [1-eps, 1+eps_high] with eps ~ 3e-4
"""

import torch
from typing import Dict

from .base_loss import BasePolicyLoss, LossOutput


class GSPOLoss(BasePolicyLoss):
    """
    GSPO: Group Sequence Policy Optimization

    Sequence-level importance sampling with tighter clipping.
    More stable than GRPO, especially for MoE models.
    """

    def __init__(self, config: dict):
        # Force sequence-level for GSPO
        config["importance_sampling_level"] = "sequence"
        super().__init__(config)

        # GSPO uses much smaller epsilon (paper recommends 3e-4 to 4e-4)
        self.epsilon = config.get("epsilon", 3e-4)
        self.epsilon_high = config.get("epsilon_high", 4e-4)

        # GSPO often uses beta=0 (no KL regularization)
        self.beta = config.get("beta", 0.0)

    def compute_sequence_importance(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sequence-level importance weight.

        w_i = exp(1/|y_i| * sum_t(log(pi_theta(y_t|...)) - log(pi_old(y_t|...))))

        This is the geometric mean of token-level ratios.
        """
        # Per-token log ratio
        log_ratio = log_probs - old_log_probs

        # Sequence lengths
        seq_lengths = mask.sum(dim=-1).clamp(min=1)  # [batch]

        # Mean log ratio per sequence (length-normalized)
        seq_log_ratio = (log_ratio * mask).sum(dim=-1) / seq_lengths  # [batch]

        # Exponentiate to get importance weight
        seq_weight = torch.exp(seq_log_ratio)  # [batch]

        return seq_weight

    def clip_sequence_ratio(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip sequence-level importance ratio.

        GSPO uses tighter clipping: [1-eps, 1+eps_high] with eps ~ 3e-4
        """
        # Asymmetric clipping based on advantage sign
        clip_high = torch.where(
            advantages > 0,
            torch.full_like(ratio, 1.0 + self.epsilon_high),
            torch.full_like(ratio, 1.0 + self.epsilon),
        )
        clip_low = torch.full_like(ratio, 1.0 - self.epsilon)

        # Use min/max operations for element-wise clipping with tensor bounds
        clipped = torch.max(ratio, clip_low)
        clipped = torch.min(clipped, clip_high)
        return clipped

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> LossOutput:
        """
        Compute GSPO loss with sequence-level operations.

        Args:
            log_probs: [batch, seq_len] current policy log probs
            old_log_probs: [batch, seq_len] old policy log probs
            ref_log_probs: [batch, seq_len] reference policy log probs
            advantages: [batch] sequence-level advantages
            mask: [batch, seq_len] attention mask
        """
        batch_size, seq_len = log_probs.shape

        # Ensure advantages are 1D (sequence-level)
        if advantages.dim() > 1:
            # Take mean if token-level advantages provided
            seq_lengths = mask.sum(dim=-1).clamp(min=1)
            advantages = (advantages * mask).sum(dim=-1) / seq_lengths

        # Compute sequence-level importance weight
        seq_weight = self.compute_sequence_importance(log_probs, old_log_probs, mask)

        # Clip sequence weight
        clipped_weight = self.clip_sequence_ratio(seq_weight, advantages)

        # Policy gradient loss
        # In GSPO, all tokens in a sequence share the same weight
        seq_lengths = mask.sum(dim=-1).clamp(min=1)

        # Per-sequence policy loss: w * A * (1/|y|) * sum_t(log_pi(y_t))
        seq_log_prob = (log_probs * mask).sum(dim=-1) / seq_lengths  # [batch]

        # Clipped objective
        surr1 = seq_weight * advantages * seq_log_prob
        surr2 = clipped_weight * advantages * seq_log_prob

        # Take min for positive advantages, max for negative
        policy_loss_per_seq = -torch.where(
            advantages > 0,
            torch.min(surr1, surr2),
            torch.max(surr1, surr2),
        )

        policy_loss = policy_loss_per_seq.mean()

        # KL divergence (optional in GSPO, often beta=0)
        kl_loss = torch.tensor(0.0, device=log_probs.device)
        if self.beta > 0:
            kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs, mask)

        # Total loss
        total_loss = policy_loss + self.beta * kl_loss

        # Metrics
        with torch.no_grad():
            clip_fraction = ((seq_weight - clipped_weight).abs() > 1e-8).float().mean()

            metrics = {
                "policy_loss": policy_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "mean_seq_weight": seq_weight.mean().item(),
                "std_seq_weight": seq_weight.std().item(),
                "clip_fraction": clip_fraction.item(),
                "mean_advantage": advantages.mean().item(),
                "seq_weight_min": seq_weight.min().item(),
                "seq_weight_max": seq_weight.max().item(),
            }

        return LossOutput(
            loss=total_loss,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            metrics=metrics,
        )
