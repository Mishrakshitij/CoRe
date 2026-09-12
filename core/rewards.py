"""CORE reward equations (2)--(7), with explicit implementation choices.

Only NumPy is required to import this module. The semantic encoder is injected
or lazily loaded; it is never a trainable member of a policy. See the paper
alignment document for underspecified defaults and the literal Eq. (3)
degeneracy when the diverse-set cap covers the entire question group.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Callable, Protocol, Sequence

import numpy as np

from .parsing import normalize_answer


class Encoder(Protocol):
    def encode(self, sentences: list[str]) -> np.ndarray: ...


class FrozenSentenceEncoder:
    """Lazy, inference-only MiniLM encoder used for Eq. (13)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name, self.device, self._model = model_name, device, None

    def encode(self, sentences: list[str]) -> np.ndarray:
        import torch
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
            self._model.requires_grad_(False)
        with torch.inference_mode():
            return np.asarray(self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False))


@dataclass
class Sample:
    question_id: str
    model_id: str
    round_id: str
    text: str
    answer: str | None
    correct: bool
    hint_provided: bool = False
    cold_succeeded: bool = False
    reward: float = 0.0
    advantage: float = 0.0
    gold: str = ""
    exploit_reward: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    # Partial scoring is an explicit callback, not hidden answer similarity.
    alpha: float = 0.0
    w_exploit: float = 1.0
    w_explore: float = 0.1
    rescue_bonus: float = 0.15
    trace_accuracy_weight: float = 0.0
    cross_model_weight: float = 0.1
    explore_margin: float = 0.15
    explore_set_cap: int = 10
    cross_margin: float = 0.15
    quality_threshold: float = 1.0
    embedding_weight: float = 0.6
    structural_weight: float = 0.4
    normalization_epsilon: float = 1e-8
    # Eq. (6) defines two models. Arithmetic mean is our N>2 extension.
    cross_aggregation: str = "mean"

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        if not isinstance(self.explore_set_cap, int) or self.explore_set_cap < 1:
            raise ValueError("explore_set_cap must be a positive integer")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if not math.isclose(self.embedding_weight + self.structural_weight, 1.0):
            raise ValueError("embedding_weight and structural_weight must sum to one")
        if self.cross_aggregation != "mean":
            raise ValueError("cross_aggregation currently supports only the documented mean extension")


@dataclass(frozen=True)
class RewardDiagnostics:
    question_id: str
    diverse_indices: tuple[int, ...]
    exploration_all_zero: bool
    exploration_stop_reason: str
    reward_mean: float
    reward_std: float
    components: dict[str, list[float]] = field(default_factory=dict)


def overlap_partial_credit(prediction: str | None, gold: str) -> float:
    """Token multiset F1: an explicit implementation choice, not a paper formula.

    The paper specifies an overlap score in [0,1] without its tokenizer or exact
    definition. This optional reference choice uses normalized answer tokens;
    substitute a dataset verifier when reproducing a particular run.
    """
    pred = Counter(re.findall(r"\w+|[^\w\s]", normalize_answer(prediction)))
    target = Counter(re.findall(r"\w+|[^\w\s]", normalize_answer(gold)))
    if not pred or not target:
        return 0.0
    return 2.0 * sum((pred & target).values()) / (sum(pred.values()) + sum(target.values()))


# The article gives operation examples but no complete regex table. These rules
# are the documented implementation, with word boundaries to avoid substring
# false positives (e.g. 'different' must not trigger the 'if' case operator).
_OPERATIONS = {
    "addition": r"\+|\b(?:add(?:ition|ing)?|sum|plus|total)\b",
    "subtraction": r"(?<=\w)\s*-\s*(?=\w)|\b(?:subtract\w*|minus|difference|remain\w*)\b",
    "multiplication": r"\*|×|\\times\b|\b(?:multipli\w*|multiply|times|product)\b",
    "division": r"/|÷|\\div\b|\b(?:divid\w*|division|split|per)\b",
    "equation": r"=|\b(?:equals?|equation)\b",
    "comparison": r"[<>≥≤]|\\(?:geq|leq)\b|\b(?:greater|less|more|fewer)\b",
    "percentage": r"%|\bpercent\w*\b",
    "fraction": r"\d\s*/\s*\d|\\d?frac\b|\bfraction\w*\b",
    "square": r"²|\^\s*\{?2\}?|\bsquar\w*\b",
    "root": r"√|\\sqrt\b|\b(?:sqrt|root)\b",
    "substitution": r"\b(?:substitut\w*|replace\w*|plug\s+in)\b",
    "simplification": r"\b(?:simplif\w*|reduce\w*|cancel\w*)\b",
    "factoring": r"\b(?:factor\w*|common)\b",
    "expansion": r"\b(?:expand\w*|distribut\w*|foil)\b",
    "isolation": r"\b(?:isolat\w*|solve\s+for|move)\b",
    "verification": r"\b(?:check\w*|verif\w*|confirm\w*)\b",
    "case_analysis": r"\b(?:cases?|if|when|scenario\w*)\b",
    "induction": r"\b(?:pattern\w*|sequence\w*|induct\w*)\b",
}


def operation_signatures(text: str) -> frozenset[str]:
    text = re.sub(r"</?[A-Za-z][^>]*>", " ", text)
    return frozenset(name for name, regex in _OPERATIONS.items() if re.search(regex, text, re.I))


def hybrid_distances(
    texts: Sequence[str], *, encoder: Encoder | None = None,
    embeddings: np.ndarray | None = None, embedding_weight: float = 0.6,
    structural_weight: float = 0.4,
) -> np.ndarray:
    """Eq. (13), one encoder call for the group and a symmetric distance matrix.

    A zero-norm embedding is rejected instead of silently assigning an invented
    cosine value. Empty operation sets have Jaccard distance zero to each other.
    Semantic cosine distance has range [0,2], not [0,1].
    """
    size = len(texts)
    if size == 0:
        return np.zeros((0, 0), dtype=float)
    if embedding_weight < 0 or structural_weight < 0 or not math.isclose(embedding_weight + structural_weight, 1):
        raise ValueError("distance weights must be nonnegative and sum to one")
    distance = np.zeros((size, size), dtype=float)
    if embedding_weight:
        if embeddings is None:
            if encoder is None:
                raise ValueError("semantic distance requires an encoder or supplied embeddings")
            embeddings = encoder.encode(list(texts))
        vectors = np.asarray(embeddings, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != size or vectors.shape[1] == 0:
            raise ValueError("embeddings must have shape (number of traces, embedding dimension)")
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must be finite")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("embedding vectors must have finite positive norms")
        unit = vectors / norms
        distance += embedding_weight * (1 - np.clip(unit @ unit.T, -1, 1))
    if structural_weight:
        operations = [operation_signatures(text) for text in texts]
        for i in range(size):
            for j in range(i):
                union = operations[i] | operations[j]
                value = 1 - len(operations[i] & operations[j]) / len(union) if union else 0.0
                distance[i, j] += structural_weight * value
                distance[j, i] += structural_weight * value
    np.fill_diagonal(distance, 0.0)
    return distance


def dpp_lite_rewards(
    distances: np.ndarray, exploit_rewards: Sequence[float], *, margin: float = 0.15,
    set_cap: int = 10,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    """Greedy max-min selection followed literally by Eq. (3).

    Start at the highest exploitation reward; ties follow input order. Admit a
    candidate iff its minimum distance is >= margin, until the cap is reached.
    Selected members receive zero; unselected members receive their distance
    beyond the margin. Consequently all rewards are zero when the cap does not
    stop selection early. This consequence is exposed, never silently repaired.
    """
    matrix = np.asarray(distances, dtype=float)
    count = len(exploit_rewards)
    if matrix.shape != (count, count) or not np.isfinite(matrix).all() or np.any(matrix < -1e-12):
        raise ValueError("distances must be a finite nonnegative square matrix aligned with rewards")
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0):
        raise ValueError("distances must be symmetric with zero diagonal")
    if not np.isfinite(exploit_rewards).all() or margin < 0 or not math.isfinite(margin):
        raise ValueError("exploitation rewards and margin must be finite; margin must be nonnegative")
    if not isinstance(set_cap, int) or set_cap < 1:
        raise ValueError("set_cap must be a positive integer")
    rewards = np.zeros(count)
    if not count:
        return rewards, (), "empty"
    selected = [int(np.argmax(exploit_rewards))]
    remaining = set(range(count)) - set(selected)
    minimum = matrix[:, selected[0]].copy()
    while remaining and len(selected) < set_cap:
        candidate = max(sorted(remaining), key=lambda i: minimum[i])
        if minimum[candidate] < margin:
            break
        selected.append(candidate)
        remaining.remove(candidate)
        minimum = np.minimum(minimum, matrix[:, candidate])
    for index in remaining:
        rewards[index] = max(0.0, float(minimum[index]) - margin)
    reason = "exhausted" if not remaining else "cap" if len(selected) >= set_cap else "margin"
    return rewards, tuple(selected), reason


def group_normalize(rewards: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("rewards must be a finite vector and epsilon must be positive")
    if not len(values):
        return values.copy()
    # Population standard deviation; all-equal and singleton groups map to zero.
    return (values - values.mean()) / (values.std(ddof=0) + epsilon)


def score_question(
    samples: Sequence[Sample], config: RewardConfig | None = None, *,
    encoder: Encoder | None = None, embeddings: np.ndarray | None = None,
    partial_credit_fn: Callable[[str | None, str], float] | None = None,
) -> RewardDiagnostics:
    """Score one question, mutating reward/advantage fields in input order.

    Eq. (3) uses the full question pool; Eq. (5)/(6) use same-round pools.
    Normalize jointly across all models and both rounds. Cold success is derived
    from observed Round-A samples, so caller metadata cannot fabricate rescues.
    """
    config = config or RewardConfig()
    if not samples:
        raise ValueError("a question group must contain at least one sample")
    question_id = samples[0].question_id
    if any(s.question_id != question_id for s in samples):
        raise ValueError("score_question accepts exactly one question_id")
    if any(s.round_id not in {"A", "B"} for s in samples):
        raise ValueError("round_id must be A or B")
    if any(not isinstance(s.correct, (bool, np.bool_)) or
           not isinstance(s.hint_provided, (bool, np.bool_)) for s in samples):
        raise ValueError("correct and hint_provided must be Boolean verifier/protocol decisions")
    if config.alpha and partial_credit_fn is None:
        raise ValueError("alpha > 0 requires an explicit partial_credit_fn; exact overlap is unspecified in the paper")
    partial = np.zeros(len(samples))
    if config.alpha:
        for i, sample in enumerate(samples):
            if not sample.gold:
                raise ValueError("partial credit requires Sample.gold")
            partial[i] = partial_credit_fn(sample.answer, sample.gold)
        if not np.isfinite(partial).all() or np.any(partial < 0) or np.any(partial > 1):
            raise ValueError("partial_credit_fn must return a finite score in [0,1]")
    correct = np.array([float(sample.correct) for sample in samples])
    exploit = correct + config.alpha * partial
    distances = None
    if config.w_explore or config.cross_model_weight:
        distances = hybrid_distances([s.text for s in samples], encoder=encoder, embeddings=embeddings,
            embedding_weight=config.embedding_weight, structural_weight=config.structural_weight)
    exploration, selected, stop_reason = np.zeros(len(samples)), (), "disabled"
    if config.w_explore:
        exploration, selected, stop_reason = dpp_lite_rewards(distances, exploit,
            margin=config.explore_margin, set_cap=config.explore_set_cap)
    cross, accuracy, rescue = np.zeros(len(samples)), np.zeros(len(samples)), np.zeros(len(samples))
    cold: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.round_id == "A":
            cold[sample.model_id].append(sample)
    for i, sample in enumerate(samples):
        same_round = [j for j, other in enumerate(samples) if other.round_id == sample.round_id]
        accuracy[i] = correct[i] * (1.0 - float(correct[same_round].mean()))
        sample.cold_succeeded = any(s.correct for s in cold[sample.model_id])
        # Require recorded A attempts and a successful *peer* in the cold pool.
        peer_success = any(s.correct and s.model_id != sample.model_id for s in samples if s.round_id == "A")
        rescue[i] = float(sample.round_id == "B" and sample.correct and sample.hint_provided
            and bool(cold[sample.model_id]) and not sample.cold_succeeded and peer_success)
        if config.cross_model_weight:
            partner_models = sorted({samples[j].model_id for j in same_round if samples[j].model_id != sample.model_id})
            partner_values = []
            for model_id in partner_models:
                peers = [j for j in same_round if samples[j].model_id == model_id]
                # Gate the partner model, then retain *all* its traces in min.
                gate = float(max(exploit[j] for j in peers) >= config.quality_threshold)
                partner_values.append(gate * max(0.0, float(distances[i, peers].min()) - config.cross_margin))
            cross[i] = float(np.mean(partner_values)) if partner_values else 0.0
    rewards = (config.w_exploit * exploit + config.w_explore * exploration
        + config.rescue_bonus * rescue + config.trace_accuracy_weight * accuracy
        + config.cross_model_weight * cross)
    advantages = group_normalize(rewards, config.normalization_epsilon)
    for i, sample in enumerate(samples):
        sample.exploit_reward, sample.reward, sample.advantage = float(exploit[i]), float(rewards[i]), float(advantages[i])
    return RewardDiagnostics(question_id, selected, bool(np.all(exploration == 0)), stop_reason,
        float(rewards.mean()), float(rewards.std(ddof=0)),
        {"exploit": exploit.tolist(), "explore": exploration.tolist(), "rescue": rescue.tolist(),
         "trace_accuracy": accuracy.tolist(), "cross_model": cross.tolist()})


def score_question_groups(
    samples: Sequence[Sample], config: RewardConfig | None = None, *,
    encoder: Encoder | None = None,
    partial_credit_fn: Callable[[str | None, str], float] | None = None,
) -> list[RewardDiagnostics]:
    """Keep independent questions separate while sharing the frozen encoder."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.question_id].append(sample)
    return [score_question(group, config, encoder=encoder, partial_credit_fn=partial_credit_fn)
            for group in groups.values()]
