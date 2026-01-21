"""
Micro-Round Manager
===================

Manages the two-phase generation process:
- Micro-round A: Cold generation (no context)
- Micro-round B: Contexted generation (with teacher hint)

Supports both standard and multi-strategy prompting modes.
"""

import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re

from ..data.preprocessing import (
    format_prompt,
    format_multi_strategy_prompt,
    format_multi_strategy_contexted_prompt,
)


@dataclass
class MicroRoundAResult:
    """Results from Micro-round A (cold generation)."""
    traces: List[str]
    rewards: List[float]
    is_correct: List[bool]
    diverse_set_indices: List[int]
    best_correct_trace: Optional[str] = None
    best_correct_idx: Optional[int] = None
    reward_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MicroRoundBResult:
    """Results from Micro-round B (contexted generation)."""
    traces: List[str]
    rewards: List[float]
    is_correct: List[bool]
    used_hint: List[bool]  # Whether each trace used the hint
    rescue_success: bool = False
    reward_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MicroRoundOutput:
    """Combined output from both micro-rounds."""
    model_id: str
    question: str
    ground_truth: str

    round_a: MicroRoundAResult
    round_b: Optional[MicroRoundBResult] = None

    teacher_context: Optional[str] = None
    teacher_source: Optional[str] = None  # Which model provided context

    # Combined traces and rewards for GRPO update
    all_traces: List[str] = field(default_factory=list)
    all_rewards: List[float] = field(default_factory=list)
    all_sources: List[str] = field(default_factory=list)  # 'cold' or 'contexted'


class MicroRoundManager:
    """
    Manages micro-round generation for collaborative training.

    Implements:
    - Cold generation (Micro-round A)
    - Context compression
    - Hint dropout
    - Contexted generation (Micro-round B)
    """

    def __init__(self, config: dict):
        self.config = config

        # Micro-round settings
        self.K = config["policy_optimization"]["K"]
        self.K_prime = config["policy_optimization"]["K_prime"]

        # Hint dropout
        self.p_hint = config["collaboration"]["p_hint"]

        # Context compression
        self.max_context_tokens = config["collaboration"]["max_context_tokens"]
        self.include_answer = config["collaboration"]["include_answer_in_context"]
        self.use_full_trace_hint = config["collaboration"].get("use_full_trace_hint", False)
        self.hint_prefix = config["collaboration"].get("hint_prefix", None)
        self.strip_answer_from_hint = config["collaboration"].get(
            "strip_answer_from_hint",
            not self.include_answer,
        )

        # Round B settings
        self.round_b_exploit_boost = config["collaboration"].get(
            "micro_round_b_exploit_boost", 0.95
        )

        # Multi-strategy prompting
        prompting_config = config.get("prompting", {})
        self.multi_strategy = prompting_config.get("multi_strategy", False)

    def compress_trace(
        self,
        trace: str,
        include_answer: bool = False,
    ) -> str:
        """
        Compress a reasoning trace to teacher context.

        Extracts:
        - Plan/approach signature
        - Key reasoning steps (1-2)
        - Optionally final answer
        """
        # Extract plan signature (operations used)
        operations = self._extract_operations(trace)
        plan_sig = " → ".join(operations) if operations else "direct computation"

        # Extract key steps
        key_steps = self._extract_key_steps(trace, max_steps=2)

        # Build compressed context
        parts = [f"Approach: {plan_sig}"]

        if key_steps:
            parts.append(f"Key insight: {key_steps}")

        if include_answer:
            answer = self._extract_answer(trace)
            if answer:
                parts.append(f"Answer: {answer}")

        context = "\n".join(parts)
        return self._truncate_context(context)

    def _build_reward_details(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Extract reward component details for logging."""
        details = []
        for res in results:
            entry: Dict[str, Any] = {}
            if hasattr(res, "total_reward"):
                entry["total"] = res.total_reward
            if hasattr(res, "exploit_reward"):
                entry["exploit"] = res.exploit_reward
            if hasattr(res, "explore_reward"):
                entry["explore"] = res.explore_reward
            if hasattr(res, "cross_reward"):
                entry["cross"] = res.cross_reward
            if hasattr(res, "think_reward"):
                entry["think"] = res.think_reward
            if hasattr(res, "rescue_bonus"):
                entry["rescue_bonus"] = res.rescue_bonus
            if hasattr(res, "correctness_reward"):
                entry["correctness"] = res.correctness_reward
            if hasattr(res, "diversity_reward"):
                entry["diversity"] = res.diversity_reward
            if hasattr(res, "consistency_reward"):
                entry["consistency"] = res.consistency_reward
            if hasattr(res, "format_reward"):
                entry["format"] = res.format_reward
            if hasattr(res, "metadata") and isinstance(res.metadata, dict):
                if "trace_acc" in res.metadata:
                    entry["trace_acc"] = res.metadata["trace_acc"]
                if "trace_acc_reward" in res.metadata:
                    entry["trace_acc_reward"] = res.metadata["trace_acc_reward"]
            details.append(entry)
        return details

    def build_teacher_context(self, trace: str) -> str:
        """Build teacher context for Round B based on config."""
        if self.use_full_trace_hint:
            context = trace
            if self.strip_answer_from_hint:
                context = self._strip_explicit_answer(context)
            return self._truncate_context(context)

        return self.compress_trace(trace, include_answer=self.include_answer)

    def _truncate_context(self, text: str) -> str:
        """Truncate context by rough token estimate (word count)."""
        words = text.split()
        if len(words) > self.max_context_tokens:
            return " ".join(words[:self.max_context_tokens])
        return text

    def _extract_operations(self, trace: str) -> List[str]:
        """Extract mathematical/reasoning operations."""
        operations = []
        trace_lower = trace.lower()

        op_patterns = [
            ("setup equation", r'equation|let\s+\w+\s*='),
            ("substitute", r'substitut|plug\s+in|replace'),
            ("simplify", r'simplif|reduc|cancel'),
            ("calculate", r'calculat|comput|multiply|divide|add|subtract'),
            ("solve", r'solve|find|get'),
            ("verify", r'check|verify|confirm'),
        ]

        for op_name, pattern in op_patterns:
            if re.search(pattern, trace_lower):
                operations.append(op_name)

        return operations[:4]  # Max 4 operations

    def _extract_key_steps(self, trace: str, max_steps: int = 2) -> str:
        """Extract key reasoning steps."""
        # Look for step markers
        step_patterns = [
            r'step\s*\d*[:\.]?\s*([^.!?\n]+[.!?])',
            r'first[,:\s]+([^.!?\n]+[.!?])',
            r'therefore[,:\s]+([^.!?\n]+[.!?])',
            r'so[,:\s]+([^.!?\n]+[.!?])',
        ]

        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, trace, re.IGNORECASE)
            steps.extend(matches)
            if len(steps) >= max_steps:
                break

        if steps:
            return " ".join(steps[:max_steps])

        # Fallback: take first substantive sentence
        sentences = re.split(r'[.!?]\s+', trace)
        for sent in sentences:
            if len(sent) > 20 and any(c.isdigit() for c in sent):
                return sent[:100]

        return ""

    def _extract_answer(self, trace: str) -> Optional[str]:
        """Extract final answer from trace."""
        patterns = [
            r'(?:the answer is|answer:|final answer:?)\s*[:\s]*([^\n]+)',
            r'\\boxed\{([^}]+)\}',
            r'####\s*([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, trace, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _strip_explicit_answer(self, trace: str) -> str:
        """Remove explicit answer lines from a trace."""
        cleaned = re.sub(r'\\boxed\{[^}]*\}', '', trace)
        cleaned = re.sub(r'(?im)^\s*(?:final answer|answer)\s*:.*$', '', cleaned)
        cleaned = re.sub(r'(?im)^\s*####\s*.*$', '', cleaned)
        lines = [line for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)

    def should_use_hint(self) -> bool:
        """Determine if hint should be used (hint dropout)."""
        return random.random() < self.p_hint

    def format_contexted_prompt(
        self,
        question: str,
        teacher_context: str,
    ) -> str:
        """
        Format prompt with teacher context for Micro-round B.

        Uses multi-strategy format if enabled.
        """
        if self.multi_strategy:
            return format_multi_strategy_contexted_prompt(question, teacher_context)
        else:
            prefix = self.hint_prefix or "Here's a helpful reasoning approach:"
            return f"""{prefix}
{teacher_context}

Now solve the following problem using a similar approach:
Question: {question}

Let's solve this step by step:"""

    def format_cold_prompt(self, question: str) -> str:
        """
        Format prompt for cold generation (Micro-round A).

        Uses multi-strategy format if enabled.
        """
        if self.multi_strategy:
            return format_multi_strategy_prompt(question)
        else:
            return format_prompt(question)

    def process_round_a(
        self,
        traces: List[str],
        ground_truth: str,
        question: str,
        reward_fn: Any,
    ) -> MicroRoundAResult:
        """
        Process Micro-round A results.

        Args:
            traces: Generated traces
            ground_truth: Correct answer
            question: Original question
            reward_fn: Reward function

        Returns:
            MicroRoundAResult
        """
        # Compute rewards
        results, diverse_indices = reward_fn.compute_batch(
            traces=traces,
            ground_truths=[ground_truth] * len(traces),
            questions=[question] * len(traces),
        )

        rewards = [r.total_reward for r in results]
        is_correct = [r.is_correct for r in results]
        reward_details = self._build_reward_details(results)

        # Find best correct trace
        best_correct_trace = None
        best_correct_idx = None
        best_correct_reward = -1

        for i, (trace, correct, reward) in enumerate(zip(traces, is_correct, rewards)):
            if correct and reward > best_correct_reward:
                best_correct_trace = trace
                best_correct_idx = i
                best_correct_reward = reward

        return MicroRoundAResult(
            traces=traces,
            rewards=rewards,
            is_correct=is_correct,
            diverse_set_indices=diverse_indices,
            best_correct_trace=best_correct_trace,
            best_correct_idx=best_correct_idx,
            reward_details=reward_details,
        )

    def process_round_b(
        self,
        traces: List[str],
        ground_truth: str,
        question: str,
        reward_fn: Any,
        used_hint: List[bool],
        round_a_had_correct: bool,
    ) -> MicroRoundBResult:
        """
        Process Micro-round B results.

        Args:
            traces: Generated traces
            ground_truth: Correct answer
            question: Original question
            reward_fn: Reward function
            used_hint: Whether each trace used teacher hint
            round_a_had_correct: Whether round A had any correct trace
        """
        # Compute rewards with rescue context
        results, _ = reward_fn.compute_batch(
            traces=traces,
            ground_truths=[ground_truth] * len(traces),
            questions=[question] * len(traces),
            trace_sources=['contexted' if h else 'cold' for h in used_hint],
            round_a_had_correct=round_a_had_correct,
        )

        rewards = [r.total_reward for r in results]
        is_correct = [r.is_correct for r in results]
        reward_details = self._build_reward_details(results)

        # Check if rescue was successful
        rescue_success = not round_a_had_correct and any(is_correct)

        return MicroRoundBResult(
            traces=traces,
            rewards=rewards,
            is_correct=is_correct,
            used_hint=used_hint,
            rescue_success=rescue_success,
            reward_details=reward_details,
        )

    def combine_rounds(
        self,
        round_a: MicroRoundAResult,
        round_b: Optional[MicroRoundBResult],
        lambda_b: float = 0.8,
    ) -> Tuple[List[str], List[float], List[str]]:
        """
        Combine traces and rewards from both rounds for GRPO update.

        Args:
            round_a: Results from Micro-round A
            round_b: Results from Micro-round B (optional)
            lambda_b: Weight for round B rewards

        Returns:
            (all_traces, all_rewards, all_sources)
        """
        all_traces = list(round_a.traces)
        all_rewards = list(round_a.rewards)
        all_sources = ['cold'] * len(round_a.traces)

        if round_b is not None:
            all_traces.extend(round_b.traces)
            # Apply lambda weighting to round B
            all_rewards.extend([r * lambda_b for r in round_b.rewards])
            all_sources.extend(
                ['contexted' if h else 'cold' for h in round_b.used_hint]
            )

        return all_traces, all_rewards, all_sources
