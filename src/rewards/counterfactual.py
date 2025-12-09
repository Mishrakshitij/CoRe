"""
Counterfactual Regularization Reward
====================================

Rewards traces that generalize to perturbed versions of the question.

R_cf(τ) = λ_cf if correct on both x and x̃, else -λ_cf/2 if correct on x but not x̃
"""

import re
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CounterfactualResult:
    """Result of counterfactual evaluation."""
    reward: float
    original_correct: bool
    counterfactual_correct: bool
    counterfactual_question: str
    perturbation_type: str


class CounterfactualReward:
    """
    Counterfactual regularization for robust reasoning strategies.

    Perturbs questions and rewards traces that generalize.
    """

    def __init__(self, config: dict):
        self.lambda_cf = config.get("lambda_cf", 0.3)
        self.penalty_factor = config.get("cf_penalty_factor", 0.5)

        self.perturbation_types = config.get(
            "perturbation_types", ["numeric", "condition"]
        )

        # Probability of applying CF evaluation
        self.cf_probability = config.get("counterfactual_probability", 0.2)

    def extract_numbers(self, text: str) -> List[Tuple[str, float, int, int]]:
        """Extract numbers with their positions."""
        pattern = r'-?\d+\.?\d*'
        matches = []
        for match in re.finditer(pattern, text):
            try:
                value = float(match.group())
                matches.append((match.group(), value, match.start(), match.end()))
            except ValueError:
                continue
        return matches

    def perturb_numeric(
        self,
        question: str,
        ground_truth: str,
    ) -> Tuple[str, str]:
        """
        Perturb numeric values in the question.

        Returns:
            (perturbed_question, new_ground_truth)
        """
        numbers = self.extract_numbers(question)

        if len(numbers) < 2:
            return question, ground_truth

        # Select a number to perturb (not the ground truth)
        gt_val = None
        try:
            gt_val = float(ground_truth)
        except ValueError:
            pass

        candidates = [
            n for n in numbers
            if gt_val is None or abs(n[1] - gt_val) > 0.01
        ]

        if not candidates:
            return question, ground_truth

        # Randomly select and perturb
        to_perturb = random.choice(candidates)
        original_str, original_val, start, end = to_perturb

        # Determine perturbation
        if original_val == 0:
            delta = random.choice([-1, 1, 2, -2])
        elif abs(original_val) < 10:
            delta = random.choice([-2, -1, 1, 2])
        else:
            delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5])

        new_val = original_val + delta

        # Format new value
        if '.' in original_str:
            new_str = f"{new_val:.{len(original_str.split('.')[1])}f}"
        else:
            new_str = str(int(new_val))

        # Construct perturbed question
        perturbed = question[:start] + new_str + question[end:]

        # Estimate new ground truth (simplified - assumes linear relationship)
        # In practice, you might need a solver or re-computation
        try:
            gt_num = float(ground_truth)
            # Simple heuristic: scale proportionally
            if original_val != 0:
                scale = new_val / original_val
                new_gt = str(int(gt_num * scale)) if gt_num == int(gt_num) else f"{gt_num * scale:.2f}"
            else:
                new_gt = ground_truth
        except ValueError:
            new_gt = ground_truth

        return perturbed, new_gt

    def perturb_condition(self, question: str) -> Tuple[str, str]:
        """
        Perturb conditions in the question.

        Swaps comparisons like "more than" <-> "less than"
        """
        swaps = [
            (r'\bmore than\b', 'less than'),
            (r'\bless than\b', 'more than'),
            (r'\bgreater than\b', 'less than'),
            (r'\bat least\b', 'at most'),
            (r'\bat most\b', 'at least'),
            (r'\bincreased\b', 'decreased'),
            (r'\bdecreased\b', 'increased'),
            (r'\badded\b', 'removed'),
            (r'\bgave away\b', 'received'),
            (r'\bbought\b', 'sold'),
            (r'\bsold\b', 'bought'),
        ]

        perturbed = question
        made_change = False

        for pattern, replacement in swaps:
            if re.search(pattern, question, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, question, count=1, flags=re.IGNORECASE)
                made_change = True
                break

        if not made_change:
            return question, "UNCHANGED"

        return perturbed, "RECOMPUTE_NEEDED"

    def generate_counterfactual(
        self,
        question: str,
        ground_truth: str,
    ) -> Tuple[str, str, str]:
        """
        Generate a counterfactual question.

        Returns:
            (perturbed_question, new_ground_truth, perturbation_type)
        """
        perturbation_type = random.choice(self.perturbation_types)

        if perturbation_type == "numeric":
            cf_question, cf_gt = self.perturb_numeric(question, ground_truth)
        elif perturbation_type == "condition":
            cf_question, cf_gt = self.perturb_condition(question)
        else:
            return question, ground_truth, "none"

        return cf_question, cf_gt, perturbation_type

    def extract_strategy(self, trace: str) -> Dict:
        """
        Extract the reasoning strategy from a trace.

        Returns a dict with operations and structure.
        """
        # Extract mathematical operations
        operations = []
        op_patterns = {
            "multiply": r'\*|×|multiply|times',
            "divide": r'/|÷|divide',
            "add": r'\+|add|plus|sum',
            "subtract": r'-|subtract|minus',
        }

        for op, pattern in op_patterns.items():
            if re.search(pattern, trace.lower()):
                operations.append(op)

        # Extract step structure
        steps = re.split(r'(?:step \d+|first|then|next|finally|therefore)', trace.lower())

        return {
            "operations": operations,
            "num_steps": len(steps),
            "has_verification": bool(re.search(r'check|verify|confirm', trace.lower())),
        }

    def apply_strategy_to_counterfactual(
        self,
        strategy: Dict,
        original_trace: str,
        cf_question: str,
        cf_ground_truth: str,
    ) -> Tuple[bool, str]:
        """
        Try to apply the same strategy to the counterfactual.

        This is a simplified version - in practice, you might use
        the model to re-generate or a symbolic solver.

        Returns:
            (is_correct, extracted_answer)
        """
        # For now, we extract the answer pattern and check structure
        # A more sophisticated version would re-execute the strategy

        # Check if the strategy would likely work
        # (heuristic based on operation presence)

        # This is a placeholder - real implementation would need
        # actual re-execution of the reasoning steps

        return False, None

    def __call__(
        self,
        trace: str,
        question: str,
        ground_truth: str,
        is_original_correct: bool,
        verify_fn=None,  # Function to verify answer
    ) -> Optional[CounterfactualResult]:
        """
        Compute counterfactual regularization reward.

        Args:
            trace: The reasoning trace
            question: Original question
            ground_truth: Original ground truth
            is_original_correct: Whether trace got original correct
            verify_fn: Optional function to verify counterfactual answer

        Returns:
            CounterfactualResult or None if CF not applied
        """
        # Probabilistic application
        if random.random() > self.cf_probability:
            return None

        # Only evaluate if original is correct
        if not is_original_correct:
            return CounterfactualResult(
                reward=0.0,
                original_correct=False,
                counterfactual_correct=False,
                counterfactual_question="",
                perturbation_type="skipped",
            )

        # Generate counterfactual
        cf_question, cf_gt, perturb_type = self.generate_counterfactual(
            question, ground_truth
        )

        if cf_gt == "UNCHANGED" or cf_question == question:
            return None

        # Check if strategy generalizes
        strategy = self.extract_strategy(trace)

        # Simple heuristic: if trace has verification step, more likely to generalize
        # In practice, would need actual re-execution
        cf_correct = False

        if verify_fn is not None:
            # Use provided verification function
            cf_correct = verify_fn(trace, cf_question, cf_gt)
        else:
            # Heuristic: strategies with verification are more robust
            cf_correct = strategy.get("has_verification", False)

        # Compute reward
        if cf_correct:
            reward = self.lambda_cf
        else:
            reward = -self.lambda_cf * self.penalty_factor

        return CounterfactualResult(
            reward=reward,
            original_correct=True,
            counterfactual_correct=cf_correct,
            counterfactual_question=cf_question,
            perturbation_type=perturb_type,
        )
