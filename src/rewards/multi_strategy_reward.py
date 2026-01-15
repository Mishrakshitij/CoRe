"""
Multi-Strategy Rewards
======================

Enhanced reward functions for multi-strategy exploration responses.
Inspired by SD-E² (Semantic Diversity with Exploit-Explore) approach.

Rewards:
- Correctness: High reward for correct final answer
- Strategy Diversity: Semantic diversity across reasoning approaches
- Outcome Consistency: Bonus when multiple strategies agree
- Format Adherence: Reward proper XML structure
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Optional: sentence-transformers for semantic diversity (lazy-loaded)
SENTENCE_MODEL = None
HAS_SENTENCE_TRANSFORMER = False

def _get_sentence_model():
    """Lazy-load sentence transformer model to avoid import-time issues."""
    global SENTENCE_MODEL, HAS_SENTENCE_TRANSFORMER
    if SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            HAS_SENTENCE_TRANSFORMER = True
        except (ImportError, Exception):
            HAS_SENTENCE_TRANSFORMER = False
    return SENTENCE_MODEL


@dataclass
class MultiStrategyRewardResult:
    """Result from multi-strategy reward computation."""
    total_reward: float
    correctness_reward: float
    diversity_reward: float
    consistency_reward: float
    format_reward: float
    num_strategies: int
    unique_strategies: int
    any_strategy_correct: bool

    @property
    def is_correct(self) -> bool:
        """Compatibility property - returns True if correctness_reward > 0."""
        return self.correctness_reward > 0


def normalize_answer(text: str) -> str:
    """Extract and normalize numerical answer for comparison."""
    if text is None:
        return ''
    text = str(text).strip()
    # Remove currency, percent, commas
    text = re.sub(r'[$£€%,]', '', text)
    # Extract numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return text.lower()


def extract_xml_final_answer(response: str) -> Optional[str]:
    """Extract final answer from XML format."""
    match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_strategy_results(response: str) -> List[str]:
    """Extract all strategy results from response."""
    results = re.findall(r'<result>\s*(.*?)\s*</result>', response, re.DOTALL)
    return [r.strip() for r in results]


def extract_strategy_reasoning(response: str) -> List[str]:
    """Extract all reasoning blocks from response."""
    reasoning = re.findall(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    return [r.strip() for r in reasoning]


def extract_strategy_approaches(response: str) -> List[str]:
    """Extract all approach names from response."""
    approaches = re.findall(r'<approach>(.*?)</approach>', response, re.DOTALL)
    return [a.strip() for a in approaches]


class MultiStrategyReward:
    """
    Multi-strategy reward function combining:
    - Exploit: Correctness reward (binary or graduated)
    - Explore: Semantic diversity across strategies
    - Consistency: Agreement bonus across strategies
    - Format: Structural compliance reward
    """

    def __init__(
        self,
        w_correct: float = 2.0,
        w_diversity: float = 0.5,
        w_consistency: float = 1.5,
        w_format: float = 0.3,
        diversity_threshold: float = 0.8,
        use_semantic_diversity: bool = True,
    ):
        """
        Initialize multi-strategy reward.

        Args:
            w_correct: Weight for correctness reward
            w_diversity: Weight for diversity reward
            w_consistency: Weight for consistency reward
            w_format: Weight for format adherence reward
            diversity_threshold: Cosine similarity threshold for "unique" strategies
            use_semantic_diversity: Use embeddings for diversity (requires sentence-transformers)
        """
        self.w_correct = w_correct
        self.w_diversity = w_diversity
        self.w_consistency = w_consistency
        self.w_format = w_format
        self.diversity_threshold = diversity_threshold
        self.use_semantic_diversity = use_semantic_diversity and HAS_SENTENCE_TRANSFORMER

    def compute_correctness_reward(
        self,
        response: str,
        ground_truth: str,
    ) -> Tuple[float, bool]:
        """
        Compute correctness reward.

        Returns:
            (reward, is_correct)
        """
        final_answer = extract_xml_final_answer(response)
        if final_answer is None:
            # Fallback to last result
            results = extract_strategy_results(response)
            final_answer = results[-1] if results else ''

        pred_norm = normalize_answer(final_answer)
        gold_norm = normalize_answer(ground_truth)

        is_correct = pred_norm == gold_norm

        # Try numerical comparison if string match fails
        if not is_correct:
            try:
                pred_num = float(pred_norm)
                gold_num = float(gold_norm)
                is_correct = abs(pred_num - gold_num) < 1e-5 * max(1, abs(gold_num))
            except (ValueError, TypeError):
                pass

        return (self.w_correct if is_correct else 0.0, is_correct)

    def compute_diversity_reward(
        self,
        response: str,
        ground_truth: str,
    ) -> Tuple[float, int, int]:
        """
        Compute semantic diversity reward across strategies.

        Returns:
            (reward, num_strategies, unique_count)
        """
        reasoning_blocks = extract_strategy_reasoning(response)
        num_strategies = len(reasoning_blocks)

        if num_strategies < 2:
            return (0.0, num_strategies, num_strategies)

        # Check if any strategy is correct (collapse to exploit if so)
        results = extract_strategy_results(response)
        gold_norm = normalize_answer(ground_truth)
        any_correct = any(normalize_answer(r) == gold_norm for r in results)

        if any_correct:
            # Collapse to fixed exploitation bonus
            return (self.w_consistency, num_strategies, num_strategies)

        sentence_model = _get_sentence_model() if self.use_semantic_diversity else None
        if sentence_model is None:
            # Fallback: simple count-based diversity
            unique_count = num_strategies  # Assume all are unique
            diversity = min(0.5, 0.15 * unique_count)
            return (diversity, num_strategies, unique_count)

        # Semantic diversity using embeddings
        try:
            embeddings = sentence_model.encode(reasoning_blocks)
            n = len(embeddings)

            # Compute pairwise cosine similarities
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                    )
                    similarities.append(sim)

            if similarities:
                mean_sim = np.mean(similarities)
                diversity = 1.0 - mean_sim

                # Count unique strategies (similarity < threshold)
                unique_count = 1
                for i in range(1, n):
                    max_sim_to_prev = max(
                        np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                        )
                        for j in range(i)
                    )
                    if max_sim_to_prev < self.diversity_threshold:
                        unique_count += 1

                # Combined: unique_count * diversity (capped)
                reward = min(self.w_diversity, 0.1 * unique_count * diversity)
                return (reward, num_strategies, unique_count)
            else:
                return (0.1, num_strategies, 1)

        except Exception:
            return (0.1, num_strategies, num_strategies)

    def compute_consistency_reward(
        self,
        response: str,
        ground_truth: str,
    ) -> Tuple[float, bool]:
        """
        Compute outcome consistency reward.
        Bonus if any strategy outcome matches ground truth.

        Returns:
            (reward, any_correct)
        """
        results = extract_strategy_results(response)
        gold_norm = normalize_answer(ground_truth)

        any_correct = any(normalize_answer(r) == gold_norm for r in results)

        return (self.w_consistency if any_correct else 0.0, any_correct)

    def compute_format_reward(self, response: str) -> float:
        """
        Compute format adherence reward.

        Rewards:
        - 0.2 per valid strategy block
        - 0.5 for having <final_answer> tag
        - Capped at w_format
        """
        pattern = re.compile(
            r'<strategy\s+id="\d+">\s*'
            r'<approach>.*?</approach>\s*'
            r'<reasoning>.*?</reasoning>\s*'
            r'<result>.*?</result>\s*'
            r'</strategy>',
            re.DOTALL
        )

        strategies = pattern.findall(response)
        has_final = '<final_answer>' in response and '</final_answer>' in response

        reward = min(1.0, 0.2 * len(strategies))
        if has_final:
            reward += 0.5

        return min(self.w_format, reward * self.w_format)

    def compute(
        self,
        response: str,
        ground_truth: str,
    ) -> MultiStrategyRewardResult:
        """
        Compute combined multi-strategy reward.

        Args:
            response: Model response text
            ground_truth: Correct answer

        Returns:
            MultiStrategyRewardResult
        """
        # Individual rewards
        correctness_reward, is_correct = self.compute_correctness_reward(response, ground_truth)
        diversity_reward, num_strategies, unique_strategies = self.compute_diversity_reward(
            response, ground_truth
        )
        consistency_reward, any_correct = self.compute_consistency_reward(response, ground_truth)
        format_reward = self.compute_format_reward(response)

        # If final answer is correct, don't double-count consistency
        if is_correct:
            consistency_reward = 0.0

        total_reward = correctness_reward + diversity_reward + consistency_reward + format_reward

        return MultiStrategyRewardResult(
            total_reward=total_reward,
            correctness_reward=correctness_reward,
            diversity_reward=diversity_reward,
            consistency_reward=consistency_reward,
            format_reward=format_reward,
            num_strategies=num_strategies,
            unique_strategies=unique_strategies,
            any_strategy_correct=any_correct or is_correct,
        )

    def compute_batch(
        self,
        traces: List[str] = None,
        responses: List[str] = None,
        ground_truths: List[str] = None,
        questions: List[str] = None,
        trace_sources: List[str] = None,
        round_a_had_correct: bool = True,
        **kwargs,
    ) -> Tuple[List[MultiStrategyRewardResult], List[int]]:
        """
        Compute rewards for a batch of responses.

        Compatible with CombinedRewardFunction interface.

        Args:
            traces: List of model responses (alias for responses)
            responses: List of model responses
            ground_truths: List of correct answers
            questions: List of questions (unused, for compatibility)
            trace_sources: Source of each trace ('cold' or 'contexted')
            round_a_had_correct: Whether round A had correct (for rescue bonus)

        Returns:
            (results, diverse_indices) tuple for compatibility
        """
        # Handle both 'traces' and 'responses' parameter names
        responses = traces if traces is not None else responses

        results = []
        for i, (response, gt) in enumerate(zip(responses, ground_truths)):
            result = self.compute(response, gt)

            # Apply rescue bonus if applicable
            is_rescue = (
                trace_sources is not None and
                i < len(trace_sources) and
                trace_sources[i] == 'contexted' and
                not round_a_had_correct
            )
            if is_rescue and result.any_strategy_correct:
                # Add rescue bonus
                result = MultiStrategyRewardResult(
                    total_reward=result.total_reward + 0.15,  # r_teach bonus
                    correctness_reward=result.correctness_reward,
                    diversity_reward=result.diversity_reward,
                    consistency_reward=result.consistency_reward,
                    format_reward=result.format_reward,
                    num_strategies=result.num_strategies,
                    unique_strategies=result.unique_strategies,
                    any_strategy_correct=result.any_strategy_correct,
                )

            results.append(result)

        # Return diverse indices (all indices for multi-strategy since diversity is internal)
        diverse_indices = list(range(len(results)))

        return results, diverse_indices


# Standalone reward functions (for compatibility with GRPO-style training)
def exact_match_reward(responses: List[str], ground_truths: List[str]) -> List[float]:
    """Binary exact match reward."""
    rewards = []
    for response, gt in zip(responses, ground_truths):
        final = extract_xml_final_answer(response) or ''
        pred_norm = normalize_answer(final)
        gold_norm = normalize_answer(gt)
        rewards.append(1.0 if pred_norm == gold_norm else 0.0)
    return rewards


def correctness_reward(responses: List[str], ground_truths: List[str]) -> List[float]:
    """Higher-weighted correctness reward."""
    return [2.0 * r for r in exact_match_reward(responses, ground_truths)]


def semantic_diversity_reward(responses: List[str], ground_truths: List[str]) -> List[float]:
    """Semantic diversity reward using embeddings."""
    reward_fn = MultiStrategyReward(w_diversity=0.5)
    results = reward_fn.compute_batch(responses, ground_truths)
    return [r.diversity_reward for r in results]


def format_adherence_reward(responses: List[str]) -> List[float]:
    """Format compliance reward."""
    reward_fn = MultiStrategyReward(w_format=1.0)
    return [reward_fn.compute_format_reward(r) for r in responses]


def combined_multi_strategy_reward(
    responses: List[str],
    ground_truths: List[str],
    w_correct: float = 2.0,
    w_diversity: float = 0.5,
    w_consistency: float = 1.5,
    w_format: float = 0.3,
) -> List[float]:
    """
    Combined reward for multi-strategy responses.

    Args:
        responses: Model responses
        ground_truths: Correct answers
        w_correct: Correctness weight
        w_diversity: Diversity weight
        w_consistency: Consistency weight
        w_format: Format weight

    Returns:
        List of total rewards
    """
    reward_fn = MultiStrategyReward(
        w_correct=w_correct,
        w_diversity=w_diversity,
        w_consistency=w_consistency,
        w_format=w_format,
    )
    results = reward_fn.compute_batch(responses, ground_truths)
    return [r.total_reward for r in results]
