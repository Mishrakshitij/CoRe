"""
Cross-Model Reward
==================

Rewards traces for complementarity with partner model's traces.

R_cross(τ) = η * min_{τ' ∈ partner(x)} d(τ, τ')

Quality-gated: only applied if R_exploit(τ) >= α * partial_thresh
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class CrossRewardResult:
    """Result of cross-model reward computation."""
    reward: float
    min_distance_to_partner: float
    quality_gated: bool
    num_partner_traces: int


class CrossModelReward:
    """
    Cross-model complementarity reward.

    Encourages models to explore different reasoning strategies
    while maintaining minimum quality.
    """

    def __init__(
        self,
        config: dict,
        distance_fn: Callable[[str, str], float] = None,
    ):
        self.eta = config.get("eta", 0.1)  # Cross-reward weight
        self.alpha = config.get("alpha", 0.3)
        self.partial_thresh = config.get("partial_thresh", 0.5)

        # Quality gate threshold
        self.quality_gate = self.alpha * self.partial_thresh

        # Distance function (shared with explore reward)
        self._distance_fn = distance_fn

        # Optional: only compare against quality partner traces
        self.filter_partner_by_quality = config.get("filter_partner_by_quality", True)

    def set_distance_fn(self, distance_fn: Callable[[str, str], float]):
        """Set the distance function."""
        self._distance_fn = distance_fn

    def compute_distance(self, trace1: str, trace2: str) -> float:
        """Compute distance between traces."""
        if self._distance_fn is None:
            raise ValueError("Distance function not set. Call set_distance_fn() first.")
        return self._distance_fn(trace1, trace2)

    def __call__(
        self,
        trace: str,
        exploit_reward: float,
        partner_traces: List[str],
        partner_exploit_rewards: List[float] = None,
    ) -> CrossRewardResult:
        """
        Compute cross-model reward for a trace.

        Args:
            trace: The trace to evaluate
            exploit_reward: This trace's exploitation reward
            partner_traces: Traces from partner model
            partner_exploit_rewards: Partner traces' exploit rewards (for filtering)

        Returns:
            CrossRewardResult with reward and metadata
        """
        # Quality gate: only apply if trace has minimum quality
        if exploit_reward < self.quality_gate:
            return CrossRewardResult(
                reward=0.0,
                min_distance_to_partner=0.0,
                quality_gated=True,
                num_partner_traces=len(partner_traces),
            )

        if not partner_traces:
            return CrossRewardResult(
                reward=0.0,
                min_distance_to_partner=0.0,
                quality_gated=False,
                num_partner_traces=0,
            )

        # Filter partner traces by quality if enabled
        if self.filter_partner_by_quality and partner_exploit_rewards is not None:
            quality_partners = [
                pt for pt, pr in zip(partner_traces, partner_exploit_rewards)
                if pr >= self.quality_gate
            ]
        else:
            quality_partners = partner_traces

        if not quality_partners:
            return CrossRewardResult(
                reward=0.0,
                min_distance_to_partner=0.0,
                quality_gated=False,
                num_partner_traces=0,
            )

        # Compute min distance to partner traces
        min_dist = min(self.compute_distance(trace, pt) for pt in quality_partners)

        # Reward: η * min_dist
        reward = self.eta * min_dist

        return CrossRewardResult(
            reward=reward,
            min_distance_to_partner=min_dist,
            quality_gated=False,
            num_partner_traces=len(quality_partners),
        )

    def batch_compute(
        self,
        traces: List[str],
        exploit_rewards: List[float],
        partner_traces: List[str],
        partner_exploit_rewards: List[float] = None,
    ) -> List[CrossRewardResult]:
        """Compute cross rewards for all traces."""
        return [
            self(trace, er, partner_traces, partner_exploit_rewards)
            for trace, er in zip(traces, exploit_rewards)
        ]


class MultiModelCrossReward:
    """
    Cross-model reward for N > 2 models.

    Computes complementarity against all partner models.
    """

    def __init__(
        self,
        config: dict,
        distance_fn: Callable[[str, str], float] = None,
    ):
        self.eta = config.get("eta", 0.1)
        self.alpha = config.get("alpha", 0.3)
        self.partial_thresh = config.get("partial_thresh", 0.5)
        self.quality_gate = self.alpha * self.partial_thresh

        self._distance_fn = distance_fn

        # Aggregation method: 'min', 'mean', 'max'
        self.aggregation = config.get("cross_aggregation", "mean")

    def set_distance_fn(self, distance_fn: Callable[[str, str], float]):
        """Set the distance function."""
        self._distance_fn = distance_fn

    def compute_distance(self, trace1: str, trace2: str) -> float:
        """Compute distance between traces."""
        if self._distance_fn is None:
            raise ValueError("Distance function not set.")
        return self._distance_fn(trace1, trace2)

    def __call__(
        self,
        trace: str,
        exploit_reward: float,
        all_partner_traces: Dict[str, List[str]],  # model_id -> traces
        all_partner_rewards: Dict[str, List[float]] = None,
    ) -> CrossRewardResult:
        """
        Compute cross-model reward against all partners.

        Args:
            trace: The trace to evaluate
            exploit_reward: This trace's exploitation reward
            all_partner_traces: Dict mapping model_id to their traces
            all_partner_rewards: Dict mapping model_id to exploit rewards

        Returns:
            CrossRewardResult
        """
        # Quality gate
        if exploit_reward < self.quality_gate:
            total_partners = sum(len(t) for t in all_partner_traces.values())
            return CrossRewardResult(
                reward=0.0,
                min_distance_to_partner=0.0,
                quality_gated=True,
                num_partner_traces=total_partners,
            )

        # Collect all partner traces (optionally quality-filtered)
        all_distances = []

        for model_id, partner_traces in all_partner_traces.items():
            if not partner_traces:
                continue

            # Filter by quality if rewards provided
            if all_partner_rewards is not None and model_id in all_partner_rewards:
                partner_rewards = all_partner_rewards[model_id]
                quality_traces = [
                    pt for pt, pr in zip(partner_traces, partner_rewards)
                    if pr >= self.quality_gate
                ]
            else:
                quality_traces = partner_traces

            if quality_traces:
                # Min distance to this partner
                min_dist = min(self.compute_distance(trace, pt) for pt in quality_traces)
                all_distances.append(min_dist)

        if not all_distances:
            return CrossRewardResult(
                reward=0.0,
                min_distance_to_partner=0.0,
                quality_gated=False,
                num_partner_traces=0,
            )

        # Aggregate across partners
        if self.aggregation == "min":
            agg_dist = min(all_distances)
        elif self.aggregation == "mean":
            agg_dist = np.mean(all_distances)
        elif self.aggregation == "max":
            agg_dist = max(all_distances)
        else:
            agg_dist = np.mean(all_distances)

        reward = self.eta * agg_dist

        total_partners = sum(len(t) for t in all_partner_traces.values())

        return CrossRewardResult(
            reward=reward,
            min_distance_to_partner=agg_dist,
            quality_gated=False,
            num_partner_traces=total_partners,
        )
