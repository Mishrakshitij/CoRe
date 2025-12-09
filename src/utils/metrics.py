"""
Metrics Utilities
=================

Compute and track evaluation metrics for collaborative reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class NoveltyMetrics:
    """
    Track novelty/diversity metrics during training.

    Metrics:
    - Pairwise distance among top traces
    - Distinct operation signatures
    - Rescue rate
    """

    pairwise_distances: List[float] = field(default_factory=list)
    subgoal_hashes: List[int] = field(default_factory=list)
    rescue_counts: Dict[str, int] = field(default_factory=lambda: {"A_failed": 0, "B_rescued": 0})
    accuracies: Dict[str, List[bool]] = field(default_factory=lambda: {"A": [], "B": []})

    def log_question(
        self,
        traces_A: List[str],
        traces_B: List[str],
        question: str,
        d_func,
        is_correct_fn,
    ):
        """Log metrics for one question."""
        # 1. Pairwise distance among top-3 traces
        all_traces = traces_A + traces_B
        if len(all_traces) >= 2:
            # Simple: compute all pairwise distances
            distances = []
            for i, t1 in enumerate(all_traces[:3]):
                for t2 in all_traces[i+1:4]:
                    try:
                        d = d_func(t1, t2)
                        distances.append(d)
                    except Exception:
                        pass

            if distances:
                self.pairwise_distances.append(np.mean(distances))

        # 2. Distinct signatures (placeholder - would need signature extraction)
        self.subgoal_hashes.append(min(len(all_traces), 5))

        # 3. Rescue tracking
        A_correct = any(is_correct_fn(t, question) for t in traces_A) if traces_A else False
        B_correct = any(is_correct_fn(t, question) for t in traces_B) if traces_B else False

        self.accuracies["A"].append(A_correct)
        self.accuracies["B"].append(B_correct)

        if not A_correct:
            self.rescue_counts["A_failed"] += 1
            if B_correct:
                self.rescue_counts["B_rescued"] += 1

    def report(self) -> Dict[str, float]:
        """Generate summary report."""
        rescue_rate = (
            self.rescue_counts["B_rescued"] /
            max(1, self.rescue_counts["A_failed"])
        )

        return {
            "avg_pairwise_dist": np.mean(self.pairwise_distances) if self.pairwise_distances else 0.0,
            "avg_distinct_signatures": np.mean(self.subgoal_hashes) if self.subgoal_hashes else 0.0,
            "rescue_rate": rescue_rate,
            "accuracy_A": np.mean(self.accuracies["A"]) if self.accuracies["A"] else 0.0,
            "accuracy_B": np.mean(self.accuracies["B"]) if self.accuracies["B"] else 0.0,
            "accuracy_either": np.mean([
                a or b for a, b in zip(self.accuracies["A"], self.accuracies["B"])
            ]) if self.accuracies["A"] else 0.0,
        }

    def reset(self):
        """Reset all metrics."""
        self.pairwise_distances = []
        self.subgoal_hashes = []
        self.rescue_counts = {"A_failed": 0, "B_rescued": 0}
        self.accuracies = {"A": [], "B": []}


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
    traces: List[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted answers
        ground_truths: Ground truth answers
        traces: Optional reasoning traces for diversity metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Accuracy
    correct = sum(
        1 for pred, gt in zip(predictions, ground_truths)
        if normalize_answer(pred) == normalize_answer(gt)
    )
    metrics["accuracy"] = correct / len(predictions) if predictions else 0.0

    # If traces provided, compute diversity
    if traces:
        # Average traces per question
        metrics["avg_traces"] = np.mean([len(t) for t in traces])

    return metrics


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""

    answer = str(answer).lower().strip()

    # Remove common formatting
    import re
    answer = re.sub(r'[$,]', '', answer)
    answer = re.sub(r'[%]$', '', answer)

    return answer


def compute_collaboration_gain(
    model_accuracies: Dict[str, float],
    combined_accuracy: float,
) -> float:
    """
    Compute collaboration gain.

    Args:
        model_accuracies: Per-model accuracy
        combined_accuracy: Combined (any correct) accuracy

    Returns:
        Collaboration gain (combined - best single)
    """
    if not model_accuracies:
        return 0.0

    best_single = max(model_accuracies.values())
    return combined_accuracy - best_single
