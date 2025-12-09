"""
Reward Functions for Collaborative Reasoning
=============================================

Implements exploration, exploitation, and cross-model rewards.

Reward Components:
- R_exploit: Correctness-based reward
- R_explore: Diversity-based reward (DPP-lite)
- R_cross: Cross-model complementarity reward
"""

from .exploit_reward import ExploitReward
from .explore_reward import ExploreReward
from .cross_reward import CrossModelReward
from .combined_reward import CombinedRewardFunction
from .counterfactual import CounterfactualReward

__all__ = [
    "ExploitReward",
    "ExploreReward",
    "CrossModelReward",
    "CombinedRewardFunction",
    "CounterfactualReward",
]
