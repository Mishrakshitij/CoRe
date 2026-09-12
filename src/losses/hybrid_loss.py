"""
GSPO-SAPO Hybrid Loss
=====================

Combines the best of GSPO and SAPO:
- GSPO: Sequence-level importance weights for stability
- SAPO: Soft gating for sample efficiency

This hybrid approach:
1. Computes sequence-level importance (GSPO style)
2. Applies soft gating at token level (SAPO style)
3. Combines both for stable yet efficient updates
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base_loss import BasePolicyLoss, LossOutput


class GSPOSAPOHybridLoss(BasePolicyLoss):
    """
    Hybrid GSPO-SAPO Loss

    Combines:
    - Sequence-level coherence from GSPO
    - Soft adaptive gating from SAPO
    - Special handling for collaborative learning scenarios
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # GSPO parameters
        self.epsilon = config.get("epsilon", 3e-4)
        self.epsilon_high = config.get("epsilon_high", 4e-4)

        # SAPO parameters
        self.tau_pos = config.get("tau_pos", 1.0)
        self.tau_neg = config.get("tau_neg", 1.05)
        self.tau_rescue = config.get("tau_rescue", 0.8)

        # Hybrid blending
        self.gspo_weight = config.get("gspo_weight", 0.5)
        self.sapo_weight = config.get("sapo_weight", 0.5)

        # KL coefficient
        self.beta = config.get("beta", 0.04)

        # Mode: 'blend', 'gspo_gate', 'sapo_seq'
        self.hybrid_mode = config.get("hybrid_mode", "gspo_gate")

    def compute_gspo_seq_weight(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GSPO-style sequence-level importance weight."""
        log_ratio = log_probs - old_log_probs
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (log_ratio * mask).sum(dim=-1) / seq_lengths
        return torch.exp(seq_log_ratio)

    def clip_gspo_weight(
        self,
        weight: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Clip sequence weight GSPO-style."""
        clip_high = torch.where(
            advantages > 0,
            torch.full_like(weight, 1.0 + self.epsilon_high),
            torch.full_like(weight, 1.0 + self.epsilon),
        )
        clip_low = torch.full_like(weight, 1.0 - self.epsilon)
        return torch.clamp(weight, clip_low, clip_high)

    def compute_sapo_gate(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        is_rescue: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SAPO-style soft gate."""
        # Select temperature
        if is_rescue is not None and is_rescue.any():
            is_rescue_expanded = is_rescue.unsqueeze(-1).expand_as(advantages) if is_rescue.dim() == 1 else is_rescue
            tau = torch.where(
                is_rescue_expanded,
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

        gate = torch.sigmoid(tau * (ratio - 1.0))
        scaling = 4.0 / tau
        return gate * scaling

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        is_rescue: Optional[torch.Tensor] = None,
        trace_source: Optional[torch.Tensor] = None,  # 0=cold, 1=contexted
    ) -> LossOutput:
        """
        Compute hybrid GSPO-SAPO loss.

        Args:
            log_probs: [batch, seq_len] current policy log probs
            old_log_probs: [batch, seq_len] old policy log probs
            ref_log_probs: [batch, seq_len] reference policy log probs
            advantages: [batch] or [batch, seq_len] advantages
            mask: [batch, seq_len] attention mask
            is_rescue: [batch] optional rescue flag
            trace_source: [batch] optional trace source (cold vs contexted)
        """
        batch_size, seq_len = log_probs.shape

        # Compute token-level ratio
        token_ratio = torch.exp(log_probs - old_log_probs)

        # Compute GSPO sequence weight
        gspo_seq_weight = self.compute_gspo_seq_weight(log_probs, old_log_probs, mask)

        # Ensure advantages are 1D for sequence-level operations
        if advantages.dim() == 1:
            seq_advantages = advantages
            advantages_expanded = advantages.unsqueeze(-1).expand(-1, seq_len)
        else:
            seq_lengths = mask.sum(dim=-1).clamp(min=1)
            seq_advantages = (advantages * mask).sum(dim=-1) / seq_lengths
            advantages_expanded = advantages

        # Clip GSPO weight
        gspo_clipped = self.clip_gspo_weight(gspo_seq_weight, seq_advantages)

        if self.hybrid_mode == "gspo_gate":
            # Mode 1: GSPO sequence weight + SAPO soft gate
            # Sequence coherence from GSPO, token adaptation from SAPO

            sapo_gate = self.compute_sapo_gate(token_ratio, advantages_expanded, is_rescue)

            # Combined weight: GSPO controls sequence, SAPO controls tokens
            # Expand GSPO weight to token level
            gspo_expanded = gspo_clipped.unsqueeze(-1).expand(-1, seq_len)

            # Multiplicative combination
            combined_weight = gspo_expanded * sapo_gate

            # Policy loss
            policy_loss_per_token = -combined_weight * advantages_expanded * log_probs
            policy_loss = (policy_loss_per_token * mask).sum() / mask.sum().clamp(min=1)

        elif self.hybrid_mode == "blend":
            # Mode 2: Blend GSPO and SAPO losses

            # GSPO loss component
            seq_lengths = mask.sum(dim=-1).clamp(min=1)
            seq_log_prob = (log_probs * mask).sum(dim=-1) / seq_lengths
            gspo_loss = -(gspo_clipped * seq_advantages * seq_log_prob).mean()

            # SAPO loss component
            sapo_gate = self.compute_sapo_gate(token_ratio, advantages_expanded, is_rescue)
            sapo_loss_per_token = -sapo_gate * advantages_expanded * log_probs
            sapo_loss = (sapo_loss_per_token * mask).sum() / mask.sum().clamp(min=1)

            # Blend
            policy_loss = self.gspo_weight * gspo_loss + self.sapo_weight * sapo_loss

        elif self.hybrid_mode == "sapo_seq":
            # Mode 3: SAPO with sequence-level coherence constraint
            # Apply SAPO but constrain by GSPO bounds

            sapo_gate = self.compute_sapo_gate(token_ratio, advantages_expanded, is_rescue)

            # Constrain: if GSPO would clip, reduce SAPO gate
            gspo_clipped_expanded = gspo_clipped.unsqueeze(-1).expand(-1, seq_len)
            clip_factor = gspo_clipped_expanded / gspo_seq_weight.unsqueeze(-1).expand(-1, seq_len).clamp(min=1e-8)

            # Apply constraint
            constrained_gate = sapo_gate * clip_factor.clamp(max=1.0)

            policy_loss_per_token = -constrained_gate * advantages_expanded * log_probs
            policy_loss = (policy_loss_per_token * mask).sum() / mask.sum().clamp(min=1)

        else:
            raise ValueError(f"Unknown hybrid_mode: {self.hybrid_mode}")

        # KL divergence
        kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs, mask)

        # Total loss
        total_loss = policy_loss + self.beta * kl_loss

        # Metrics
        with torch.no_grad():
            sapo_gate = self.compute_sapo_gate(token_ratio, advantages_expanded, is_rescue)

            metrics = {
                "policy_loss": policy_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "gspo_seq_weight_mean": gspo_seq_weight.mean().item(),
                "gspo_seq_weight_std": gspo_seq_weight.std().item(),
                "gspo_clip_fraction": ((gspo_seq_weight - gspo_clipped).abs() > 1e-8).float().mean().item(),
                "sapo_gate_mean": (sapo_gate * mask).sum().item() / mask.sum().item(),
                "mean_token_ratio": (token_ratio * mask).sum().item() / mask.sum().item(),
                "mean_advantage": seq_advantages.mean().item(),
                "hybrid_mode": self.hybrid_mode,
            }

            if is_rescue is not None:
                metrics["rescue_fraction"] = is_rescue.float().mean().item()

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
        is_rescue: Optional[torch.Tensor] = None,
        trace_source: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Forward with optional collaborative flags."""
        return self.compute_loss(
            log_probs, old_log_probs, ref_log_probs,
            advantages, mask, is_rescue, trace_source
        )
