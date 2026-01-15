"""
Loss Functions for Collaborative Reasoning
==========================================

Implements GRPO, GSPO, SAPO, and hybrid variants for policy optimization.

References:
- GRPO: DeepSeek-R1 (2024)
- GSPO: arxiv.org/abs/2507.18071 (Qwen Team)
- SAPO: arxiv.org/abs/2511.20347 (Qwen Team)
"""

from .grpo_loss import GRPOLoss
from .gspo_loss import GSPOLoss
from .sapo_loss import SAPOLoss
from .hybrid_loss import GSPOSAPOHybridLoss
from .base_loss import BasePolicyLoss, LossOutput

__all__ = [
    "GRPOLoss",
    "GSPOLoss",
    "SAPOLoss",
    "GSPOSAPOHybridLoss",
    "BasePolicyLoss",
    "LossOutput",
]


def get_loss_fn(algorithm: str, config: dict):
    """Factory function to get loss function by name."""
    loss_map = {
        "grpo": GRPOLoss,
        "gspo": GSPOLoss,
        "sapo": SAPOLoss,
        "gspo_sapo_hybrid": GSPOSAPOHybridLoss,
        "hybrid": GSPOSAPOHybridLoss,
    }

    if algorithm not in loss_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(loss_map.keys())}")

    return loss_map[algorithm](config)
