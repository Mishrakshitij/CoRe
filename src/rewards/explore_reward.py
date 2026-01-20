"""
Exploration Reward (DPP-Lite)
=============================

Rewards traces for diversity using greedy DPP approximation.

R_explore(τ) = max(0, min_{τ' ∈ S} d(τ, τ') - δ)

Where S is a greedily constructed diverse set starting with best exploit trace.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import re


@dataclass
class ExploreResult:
    """Result of exploration reward computation."""
    reward: float
    min_distance: float
    in_diverse_set: bool
    diverse_set_size: int


class ExploreReward:
    """
    Exploration reward using DPP-lite (greedy diverse set construction).

    Algorithm:
    1. Start with highest-exploit trace as first element of S
    2. Iteratively add trace that maximizes min distance to S
    3. Stop when min distance < δ
    4. Reward non-S traces based on distance to S
    """

    def __init__(self, config: dict, distance_fn: Callable = None):
        self.delta = config.get("delta", 0.15)  # Margin threshold
        self.max_diverse_set_size = config.get("max_diverse_set_size", 10)

        # Distance function type
        self.distance_type = config.get("distance_type", "hybrid")

        # Weights for hybrid distance
        self.embedding_weight = config.get("embedding_weight", 0.6)
        self.structural_weight = config.get("structural_weight", 0.4)

        # Custom distance function if provided
        self._custom_distance_fn = distance_fn

        # Embedding model for semantic distance
        self._embedding_model = None
        self.embedding_model_name = config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Device for embedding model (default: cuda:0 or first available GPU)
        # Set via config: distance.embedding_device: "cuda:3"
        self.embedding_device = config.get("embedding_device", None)

    @property
    def embedding_model(self):
        """Lazy load embedding model with optional device specification."""
        if self._embedding_model is None:
            if self.embedding_device:
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name, device=self.embedding_device
                )
            else:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def extract_operations(self, trace: str) -> List[str]:
        """Extract mathematical/reasoning operations from trace."""
        operations = []

        # Mathematical operations
        math_patterns = {
            "addition": r'\+|add|sum|plus|total',
            "subtraction": r'-|subtract|minus|difference|remain',
            "multiplication": r'\*|×|multiply|times|product',
            "division": r'/|÷|divide|split|per',
            "equation": r'=|equals|is equal',
            "comparison": r'>|<|≥|≤|greater|less|more|fewer',
            "percentage": r'%|percent',
            "fraction": r'\d+/\d+|fraction',
            "square": r'square|²|\^2',
            "root": r'sqrt|root|√',
        }

        trace_lower = trace.lower()
        for op_name, pattern in math_patterns.items():
            if re.search(pattern, trace_lower):
                operations.append(op_name)

        # Reasoning patterns
        reasoning_patterns = {
            "substitution": r'substitut|replace|plug in',
            "simplification": r'simplif|reduce|cancel',
            "factoring": r'factor|common',
            "expansion": r'expand|distribute|foil',
            "isolation": r'isolat|solve for|move',
            "verification": r'check|verify|confirm',
            "case_analysis": r'case|if|when|scenario',
            "induction": r'pattern|sequence|induct',
        }

        for op_name, pattern in reasoning_patterns.items():
            if re.search(pattern, trace_lower):
                operations.append(op_name)

        return operations

    def extract_plan_signature(self, trace: str) -> str:
        """Extract high-level plan signature from trace."""
        ops = self.extract_operations(trace)
        return "→".join(sorted(set(ops)))

    def compute_embedding_distance(
        self,
        trace1: str,
        trace2: str,
    ) -> float:
        """Compute semantic distance using embeddings."""
        embeddings = self.embedding_model.encode([trace1, trace2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        return 1.0 - similarity  # Convert to distance

    def compute_structural_distance(
        self,
        trace1: str,
        trace2: str,
    ) -> float:
        """Compute structural distance based on operations."""
        ops1 = set(self.extract_operations(trace1))
        ops2 = set(self.extract_operations(trace2))

        if not ops1 and not ops2:
            return 0.0
        if not ops1 or not ops2:
            return 1.0

        # Jaccard distance
        intersection = len(ops1 & ops2)
        union = len(ops1 | ops2)
        similarity = intersection / union
        return 1.0 - similarity

    def compute_distance(
        self,
        trace1: str,
        trace2: str,
    ) -> float:
        """Compute distance between two traces."""
        if self._custom_distance_fn is not None:
            return self._custom_distance_fn(trace1, trace2)

        if self.distance_type == "embedding":
            return self.compute_embedding_distance(trace1, trace2)
        elif self.distance_type == "structural":
            return self.compute_structural_distance(trace1, trace2)
        elif self.distance_type == "hybrid":
            emb_dist = self.compute_embedding_distance(trace1, trace2)
            struct_dist = self.compute_structural_distance(trace1, trace2)
            return (
                self.embedding_weight * emb_dist +
                self.structural_weight * struct_dist
            )
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")

    def build_diverse_set_dpp_lite(
        self,
        traces: List[str],
        exploit_rewards: List[float],
    ) -> Tuple[List[int], List[str]]:
        """
        Build diverse set using greedy DPP-lite algorithm.

        Args:
            traces: List of reasoning traces
            exploit_rewards: Exploitation rewards for ordering

        Returns:
            (indices, traces) of diverse set
        """
        if not traces:
            return [], []

        n = len(traces)

        # Start with highest exploit trace
        best_idx = int(np.argmax(exploit_rewards))
        S_indices = [best_idx]
        S_traces = [traces[best_idx]]

        # Pre-compute all pairwise distances (for efficiency)
        # In practice, compute lazily for large sets
        remaining = set(range(n)) - {best_idx}

        while len(S_indices) < self.max_diverse_set_size and remaining:
            best_candidate = None
            best_min_dist = -1

            for idx in remaining:
                # Compute min distance to current set
                min_dist = min(
                    self.compute_distance(traces[idx], s_trace)
                    for s_trace in S_traces
                )

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = idx

            # Stop if best candidate is below threshold
            if best_candidate is None or best_min_dist < self.delta:
                break

            S_indices.append(best_candidate)
            S_traces.append(traces[best_candidate])
            remaining.remove(best_candidate)

        return S_indices, S_traces

    def __call__(
        self,
        trace: str,
        diverse_set: List[str],
        trace_in_set: bool = False,
    ) -> ExploreResult:
        """
        Compute exploration reward for a single trace.

        Args:
            trace: The trace to evaluate
            diverse_set: The constructed diverse set S
            trace_in_set: Whether this trace is in S

        Returns:
            ExploreResult with reward and metadata
        """
        if not diverse_set:
            return ExploreResult(
                reward=0.0,
                min_distance=0.0,
                in_diverse_set=trace_in_set,
                diverse_set_size=0,
            )

        # Traces in S don't get exploration reward (they define diversity)
        if trace_in_set:
            return ExploreResult(
                reward=0.0,
                min_distance=0.0,
                in_diverse_set=True,
                diverse_set_size=len(diverse_set),
            )

        # Compute min distance to diverse set
        min_dist = min(self.compute_distance(trace, s) for s in diverse_set)

        # Reward: max(0, min_dist - δ)
        reward = max(0.0, min_dist - self.delta)

        return ExploreResult(
            reward=reward,
            min_distance=min_dist,
            in_diverse_set=False,
            diverse_set_size=len(diverse_set),
        )

    def batch_compute(
        self,
        traces: List[str],
        exploit_rewards: List[float],
    ) -> Tuple[List[ExploreResult], List[int]]:
        """
        Compute exploration rewards for all traces.

        Args:
            traces: List of reasoning traces
            exploit_rewards: Exploitation rewards for DPP-lite ordering

        Returns:
            (results, diverse_set_indices)
        """
        # Build diverse set
        S_indices, S_traces = self.build_diverse_set_dpp_lite(traces, exploit_rewards)
        S_set = set(S_indices)

        # Compute rewards for all traces
        results = []
        for i, trace in enumerate(traces):
            in_set = i in S_set
            result = self(trace, S_traces, trace_in_set=in_set)
            results.append(result)

        return results, S_indices
