"""
Think Reward for Mistral Reasoning Models
==========================================

Rewards diversity in [THINK]...[/THINK] reasoning blocks.

R_think(τ) = diversity score based on embedding distance of [THINK] content

This reward encourages the model to explore different reasoning paths
in its internal thinking process.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from ..data.preprocessing import extract_think_content


@dataclass
class ThinkRewardResult:
    """Result of think reward computation."""
    reward: float
    think_content: Optional[str]
    has_think_block: bool
    diversity_score: float


class ThinkReward:
    """
    Diversity reward for [THINK] reasoning blocks in Mistral models.

    Computes diversity across [THINK] content from multiple traces,
    encouraging the model to explore different reasoning paths.
    """

    def __init__(self, config: dict):
        self.config = config
        self.w_think = config.get("w_think", 0.1)
        self.min_think_length = config.get("min_think_length", 50)

        # Embedding model for semantic distance
        self._embedding_model = None
        self.embedding_model_name = config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Device for embedding model (default: cuda:0 or first available GPU)
        # Set via config: rewards.embedding_device: "cuda:3"
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

    def compute_embedding_distance(self, text1: str, text2: str) -> float:
        """Compute semantic distance using embeddings."""
        if not text1 or not text2:
            return 1.0

        embeddings = self.embedding_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )
        return 1.0 - similarity  # Convert to distance

    def compute_pairwise_diversity(self, think_contents: List[str]) -> float:
        """
        Compute average pairwise diversity of [THINK] contents.

        Args:
            think_contents: List of [THINK] content strings

        Returns:
            Average pairwise distance (0 to 1, higher = more diverse)
        """
        valid_contents = [c for c in think_contents if c and len(c) >= self.min_think_length]

        if len(valid_contents) < 2:
            return 0.0

        # Compute embeddings for all contents
        embeddings = self.embedding_model.encode(valid_contents)

        # Compute pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                distances.append(1.0 - similarity)

        return np.mean(distances) if distances else 0.0

    def __call__(
        self,
        trace: str,
        all_traces: List[str],
    ) -> ThinkRewardResult:
        """
        Compute think reward for a single trace.

        Args:
            trace: The trace to evaluate
            all_traces: All traces in the batch (for diversity computation)

        Returns:
            ThinkRewardResult with reward and metadata
        """
        # Extract [THINK] content from this trace
        think_content = extract_think_content(trace)
        has_think_block = think_content is not None and len(think_content) >= self.min_think_length

        if not has_think_block:
            return ThinkRewardResult(
                reward=0.0,
                think_content=None,
                has_think_block=False,
                diversity_score=0.0,
            )

        # Extract [THINK] content from all traces
        all_think_contents = [extract_think_content(t) for t in all_traces]

        # Compute diversity score
        diversity_score = self.compute_pairwise_diversity(all_think_contents)

        # Reward is weighted diversity
        reward = self.w_think * diversity_score

        return ThinkRewardResult(
            reward=reward,
            think_content=think_content,
            has_think_block=True,
            diversity_score=diversity_score,
        )

    def batch_compute(
        self,
        traces: List[str],
    ) -> Tuple[List[ThinkRewardResult], float]:
        """
        Compute think rewards for all traces in a batch.

        Args:
            traces: List of reasoning traces

        Returns:
            (results, overall_diversity_score)
        """
        # Extract all [THINK] contents
        think_contents = [extract_think_content(t) for t in traces]

        # Compute overall diversity
        overall_diversity = self.compute_pairwise_diversity(think_contents)

        # Compute individual results
        results = []
        for i, trace in enumerate(traces):
            think_content = think_contents[i]
            has_think_block = think_content is not None and len(think_content) >= self.min_think_length

            if has_think_block:
                # Reward based on overall diversity
                reward = self.w_think * overall_diversity
            else:
                reward = 0.0

            results.append(ThinkRewardResult(
                reward=reward,
                think_content=think_content,
                has_think_block=has_think_block,
                diversity_score=overall_diversity if has_think_block else 0.0,
            ))

        return results, overall_diversity
