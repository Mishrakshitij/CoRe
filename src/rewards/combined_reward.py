"""
Combined Reward Function
========================

Combines exploit, explore, and cross-model rewards with configurable weights.

R(τ) = w_explt * R_exploit(τ) + w_exp * R_explore(τ) + w_cross * R_cross(τ)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .exploit_reward import ExploitReward, ExploitResult
from .explore_reward import ExploreReward, ExploreResult
from .cross_reward import CrossModelReward, CrossRewardResult, MultiModelCrossReward
from .think_reward import ThinkReward, ThinkRewardResult


@dataclass
class CombinedRewardResult:
    """Result of combined reward computation."""
    total_reward: float
    exploit_reward: float
    explore_reward: float
    cross_reward: float
    rescue_bonus: float
    is_correct: bool
    exploit_result: ExploitResult
    explore_result: Optional[ExploreResult] = None
    cross_result: Optional[CrossRewardResult] = None
    think_reward: float = 0.0
    think_result: Optional[ThinkRewardResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CombinedRewardFunction:
    """
    Combined reward function for collaborative reasoning.

    Handles:
    - Epoch-based weight annealing
    - Rescue bonus computation
    - Multi-model cross rewards
    """

    def __init__(self, config: dict):
        self.config = config

        # Initialize component reward functions
        self.exploit_reward = ExploitReward(config)
        self.explore_reward = ExploreReward(config)

        # Cross reward (will set distance fn later)
        self.cross_reward = CrossModelReward(config)
        self.multi_cross_reward = MultiModelCrossReward(config)

        # Share distance function
        self.cross_reward.set_distance_fn(self.explore_reward.compute_distance)
        self.multi_cross_reward.set_distance_fn(self.explore_reward.compute_distance)

        # Think reward (optional, for Mistral reasoning models)
        self.use_think_reward = config.get("use_think_reward", False)
        self.think_reward_fn = ThinkReward(config) if self.use_think_reward else None
        self.w_think = config.get("w_think", 0.1)

        # Trace-accuracy reward (fraction correct across K traces)
        self.use_trace_acc_reward = config.get("use_trace_acc_reward", False)
        self.w_trace_acc = config.get("w_trace_acc", 0.0)
        self.trace_acc_apply_to = config.get("trace_acc_apply_to", "all")

        # Weights for epoch 1
        self.w_exploit_e1 = config.get("w_exploit_e1", 1.0)
        self.w_explore_e1 = config.get("w_explore_e1", 0.2)
        self.w_cross_e1 = config.get("w_cross_e1", 0.0)

        # Weights for epoch 2
        self.w_exploit_e2 = config.get("w_exploit_e2", 1.0)
        self.w_explore_e2 = config.get("w_explore_e2", 0.05)
        self.w_cross_e2 = config.get("w_cross_e2", 0.1)

        # Rescue bonus
        self.r_teach = config.get("r_teach", 0.15)

        # Current epoch (updated during training)
        self.current_epoch = 1

    def set_epoch(self, epoch: int):
        """Set current epoch for weight annealing."""
        self.current_epoch = epoch

    def get_weights(self) -> Tuple[float, float, float]:
        """Get current weights based on epoch."""
        if self.current_epoch == 1:
            return self.w_exploit_e1, self.w_explore_e1, self.w_cross_e1
        else:
            return self.w_exploit_e2, self.w_explore_e2, self.w_cross_e2

    def compute_single(
        self,
        trace: str,
        ground_truth: str,
        question: str,
        diverse_set: List[str],
        trace_in_diverse_set: bool,
        partner_traces: List[str] = None,
        partner_exploit_rewards: List[float] = None,
        is_rescue: bool = False,
        round_a_had_correct: bool = True,
        all_traces: List[str] = None,
    ) -> CombinedRewardResult:
        """
        Compute combined reward for a single trace.

        Args:
            trace: Reasoning trace
            ground_truth: Correct answer
            question: Original question
            diverse_set: DPP-lite diverse set
            trace_in_diverse_set: Whether trace is in diverse set
            partner_traces: Partner model's traces
            partner_exploit_rewards: Partner exploit rewards
            is_rescue: Whether this is a rescue trace (round B after A failed)
            round_a_had_correct: Whether round A had any correct trace
            all_traces: All traces in batch (for think reward diversity)
        """
        w_exploit, w_explore, w_cross = self.get_weights()

        # Exploitation reward
        exploit_result = self.exploit_reward(trace, ground_truth, question)

        # Exploration reward
        explore_result = self.explore_reward(
            trace, diverse_set, trace_in_diverse_set
        )

        # Cross-model reward
        cross_result = None
        cross_reward_val = 0.0
        if partner_traces and w_cross > 0:
            cross_result = self.cross_reward(
                trace,
                exploit_result.reward,
                partner_traces,
                partner_exploit_rewards,
            )
            cross_reward_val = cross_result.reward

        # Think reward (optional, for Mistral reasoning models)
        think_result = None
        think_reward_val = 0.0
        if self.use_think_reward and self.think_reward_fn and all_traces:
            think_result = self.think_reward_fn(trace, all_traces)
            think_reward_val = think_result.reward

        # Rescue bonus
        rescue_bonus = 0.0
        if is_rescue and not round_a_had_correct and exploit_result.is_correct:
            rescue_bonus = self.r_teach

        # Combined reward
        total = (
            w_exploit * exploit_result.reward +
            w_explore * explore_result.reward +
            w_cross * cross_reward_val +
            think_reward_val +
            rescue_bonus
        )

        return CombinedRewardResult(
            total_reward=total,
            exploit_reward=exploit_result.reward,
            explore_reward=explore_result.reward,
            cross_reward=cross_reward_val,
            rescue_bonus=rescue_bonus,
            is_correct=exploit_result.is_correct,
            exploit_result=exploit_result,
            explore_result=explore_result,
            cross_result=cross_result,
            think_reward=think_reward_val,
            think_result=think_result,
            metadata={
                "w_exploit": w_exploit,
                "w_explore": w_explore,
                "w_cross": w_cross,
                "w_think": self.w_think if self.use_think_reward else 0.0,
                "epoch": self.current_epoch,
                "is_rescue": is_rescue,
                "use_think_reward": self.use_think_reward,
                "trace_acc": 0.0,
                "trace_acc_reward": 0.0,
            },
        )

    def compute_batch(
        self,
        traces: List[str],
        ground_truths: List[str],
        questions: List[str],
        partner_traces: List[str] = None,
        partner_exploit_rewards: List[float] = None,
        trace_sources: List[str] = None,  # 'cold' or 'contexted'
        round_a_had_correct: bool = True,
    ) -> Tuple[List[CombinedRewardResult], List[int]]:
        """
        Compute rewards for a batch of traces from the same question.

        Args:
            traces: List of reasoning traces
            ground_truths: List of ground truths (same for all)
            questions: List of questions (same for all)
            partner_traces: Partner model's traces
            partner_exploit_rewards: Partner exploit rewards
            trace_sources: Source of each trace ('cold' or 'contexted')
            round_a_had_correct: Whether round A had correct trace

        Returns:
            (results, diverse_set_indices)
        """
        # First compute all exploit rewards
        exploit_results = self.exploit_reward.batch_compute(
            traces, ground_truths, questions
        )
        exploit_rewards = [r.reward for r in exploit_results]

        # Trace-accuracy reward (fraction correct across traces)
        trace_acc = 0.0
        trace_acc_reward = 0.0
        if self.use_trace_acc_reward and exploit_results:
            correct_count = sum(1 for r in exploit_results if r.is_correct)
            trace_acc = correct_count / max(1, len(exploit_results))
            trace_acc_reward = self.w_trace_acc * trace_acc

        # Build diverse set using DPP-lite
        explore_results, diverse_set_indices = self.explore_reward.batch_compute(
            traces, exploit_rewards
        )

        # Get diverse set traces
        diverse_set = [traces[i] for i in diverse_set_indices]

        # Compute think rewards for all traces (if enabled)
        think_results = None
        if self.use_think_reward and self.think_reward_fn:
            think_results, _ = self.think_reward_fn.batch_compute(traces)

        # Compute combined rewards
        results = []
        for i, (trace, exploit_res, explore_res) in enumerate(
            zip(traces, exploit_results, explore_results)
        ):
            is_rescue = (
                trace_sources is not None and
                trace_sources[i] == 'contexted' and
                not round_a_had_correct
            )

            w_exploit, w_explore, w_cross = self.get_weights()

            # Cross reward
            cross_result = None
            cross_reward_val = 0.0
            if partner_traces and w_cross > 0:
                cross_result = self.cross_reward(
                    trace,
                    exploit_res.reward,
                    partner_traces,
                    partner_exploit_rewards,
                )
                cross_reward_val = cross_result.reward

            # Think reward (if enabled)
            think_result = None
            think_reward_val = 0.0
            if think_results:
                think_result = think_results[i]
                think_reward_val = think_result.reward

            # Rescue bonus
            rescue_bonus = 0.0
            if is_rescue and exploit_res.is_correct:
                rescue_bonus = self.r_teach

            # Trace-accuracy reward (apply to all or correct-only)
            trace_acc_bonus = 0.0
            if self.use_trace_acc_reward and trace_acc_reward != 0.0:
                if self.trace_acc_apply_to == "correct" and not exploit_res.is_correct:
                    trace_acc_bonus = 0.0
                else:
                    trace_acc_bonus = trace_acc_reward

            total = (
                w_exploit * exploit_res.reward +
                w_explore * explore_res.reward +
                w_cross * cross_reward_val +
                think_reward_val +
                rescue_bonus +
                trace_acc_bonus
            )

            results.append(CombinedRewardResult(
                total_reward=total,
                exploit_reward=exploit_res.reward,
                explore_reward=explore_res.reward,
                cross_reward=cross_reward_val,
                rescue_bonus=rescue_bonus,
                is_correct=exploit_res.is_correct,
                exploit_result=exploit_res,
                explore_result=explore_res,
                cross_result=cross_result,
                think_reward=think_reward_val,
                think_result=think_result,
                metadata={
                    "trace_source": trace_sources[i] if trace_sources else "unknown",
                    "is_rescue": is_rescue,
                    "use_think_reward": self.use_think_reward,
                    "trace_acc": trace_acc,
                    "trace_acc_reward": trace_acc_bonus,
                },
            ))

        return results, diverse_set_indices

    def compute_multi_model(
        self,
        model_traces: Dict[str, List[str]],  # model_id -> traces
        ground_truth: str,
        question: str,
        round_a_correct_models: List[str] = None,
    ) -> Dict[str, Tuple[List[CombinedRewardResult], List[int]]]:
        """
        Compute rewards for multi-model collaboration.

        Args:
            model_traces: Dict mapping model_id to list of traces
            ground_truth: Correct answer
            question: Original question
            round_a_correct_models: Models that had correct in round A

        Returns:
            Dict mapping model_id to (results, diverse_set_indices)
        """
        round_a_correct_models = round_a_correct_models or []

        # First pass: compute exploit rewards for all
        all_exploit_rewards = {}
        for model_id, traces in model_traces.items():
            exploit_results = self.exploit_reward.batch_compute(
                traces, [ground_truth] * len(traces), [question] * len(traces)
            )
            all_exploit_rewards[model_id] = [r.reward for r in exploit_results]

        # Second pass: compute full rewards with cross-model
        results = {}
        for model_id, traces in model_traces.items():
            # Get partner traces (all other models)
            partner_traces_dict = {
                mid: model_traces[mid]
                for mid in model_traces if mid != model_id
            }
            partner_rewards_dict = {
                mid: all_exploit_rewards[mid]
                for mid in all_exploit_rewards if mid != model_id
            }

            # Flatten for cross reward
            flat_partner_traces = []
            flat_partner_rewards = []
            for mid in partner_traces_dict:
                flat_partner_traces.extend(partner_traces_dict[mid])
                flat_partner_rewards.extend(partner_rewards_dict[mid])

            # Check if round A had correct for this model
            round_a_had_correct = model_id in round_a_correct_models

            # Compute batch rewards
            batch_results, diverse_indices = self.compute_batch(
                traces=traces,
                ground_truths=[ground_truth] * len(traces),
                questions=[question] * len(traces),
                partner_traces=flat_partner_traces,
                partner_exploit_rewards=flat_partner_rewards,
                round_a_had_correct=round_a_had_correct,
            )

            results[model_id] = (batch_results, diverse_indices)

        return results
