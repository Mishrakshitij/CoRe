"""
Reward Functions for Collaborative Reasoning
=============================================

Implements exploration, exploitation, and cross-model rewards.

Reward Components:
- R_exploit: Correctness-based reward
- R_explore: Diversity-based reward (DPP-lite)
- R_cross: Cross-model complementarity reward
- Multi-Strategy: Enhanced rewards for multi-strategy exploration format
"""

from .exploit_reward import ExploitReward
from .explore_reward import ExploreReward
from .cross_reward import CrossModelReward
from .combined_reward import CombinedRewardFunction
from .counterfactual import CounterfactualReward
from .multi_strategy_reward import (
    MultiStrategyReward,
    MultiStrategyRewardResult,
    exact_match_reward,
    correctness_reward,
    semantic_diversity_reward,
    format_adherence_reward,
    combined_multi_strategy_reward,
)

__all__ = [
    "ExploitReward",
    "ExploreReward",
    "CrossModelReward",
    "CombinedRewardFunction",
    "CounterfactualReward",
    # Multi-strategy rewards
    "MultiStrategyReward",
    "MultiStrategyRewardResult",
    "exact_match_reward",
    "correctness_reward",
    "semantic_diversity_reward",
    "format_adherence_reward",
    "combined_multi_strategy_reward",
]
