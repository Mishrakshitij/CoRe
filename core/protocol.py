"""Teacher selection and answer-stripped prompts for CORE Algorithm 1.

Generation and optimization stay in the trainer. Keeping these pure helpers
independent makes selection and rescue metadata testable without model weights.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .parsing import _ANSWER_TAG, answers_equal, boxed_spans, extract_strategies, select_correct_strategy
from .rewards import Sample


def select_teacher(samples: Sequence[Sample], score: str = "reward") -> Sample | None:
    """Highest-scoring correct cold trace; stable input-order tie breaking.

    Section 4 uses total reward (the default here). Appendix A.1 instead says
    highest exploitation reward; pass ``score='exploit_reward'`` for that choice.
    """
    if score not in {"reward", "exploit_reward"}:
        raise ValueError("teacher score must be reward or exploit_reward")
    if len({sample.question_id for sample in samples}) > 1:
        raise ValueError("teacher candidates must belong to one question")
    candidates = [sample for sample in samples if sample.round_id == "A" and sample.correct]
    return max(candidates, key=lambda sample: getattr(sample, score), default=None)


def strip_explicit_answers(text: str) -> str:
    """Remove answer payloads, not just labels, before exposing a teacher hint.

    Arithmetic intermediate values can still imply the answer. This routine
    removes the paper's explicit leakage formats; it is not an information-flow
    proof or a claim that a complete correct derivation conceals its conclusion.
    """
    text = _ANSWER_TAG.sub("", text)
    # Remove nested boxes from the outside in to preserve valid span offsets.
    spans = boxed_spans(text)
    outer = [(start, end) for start, end, _ in spans
             if not any(other_start < start and other_end >= end for other_start, other_end, _ in spans)]
    for start, end in reversed(outer):
        text = text[:start] + text[end:]
    text = re.sub(
        r"(?im)^\s*(?:#{4}|(?:\*\*)?(?:(?:final\s+)?answer|outcome|result)"
        r"(?:\*\*)?\s*(?::|=|\bis\b))[^\n]*", "", text,
    )
    text = re.sub(r"(?i)\bfinal\s+answer\s*(?::|=|\bis\b)[^\n]*", "", text)
    # A truncated/mismatched explicit answer tag is unsafe as hint content.
    text = re.sub(r"(?is)<(?:final[ _-]?answer|answer|outcome|result|strategy_[\w-]+_outcome)\b[^>]*>.*", "", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()


def build_hint(
    teacher_text: str, gold: str, token_budget: int = 1536, tokenizer: Any = None,
) -> str:
    """Select a correct strategy, strip explicit outcomes and apply a length cap.

    If structured strategies exist, only a strategy with a verified answer can
    become a hint. An unstructured correct trace is used in full. With a
    tokenizer the cap is exact; without one we use the explicitly approximate
    four-characters-per-token fallback. The trainer passes the student's
    tokenizer and retains the resulting prompt verbatim for policy updates.
    """
    if not isinstance(token_budget, int) or token_budget < 0:
        raise ValueError("token_budget must be a nonnegative integer")
    if token_budget == 0:
        return ""
    strategies = extract_strategies(teacher_text)
    if strategies:
        source = select_correct_strategy(teacher_text, gold)
        if source is None:
            return ""
    else:
        source = teacher_text
    hint = strip_explicit_answers(source)
    # Bare numeric/MCQ answers are valid evaluator outputs but contain no
    # reasoning. They cannot become teacher hints or trigger a rescue bonus.
    if not hint or answers_equal(hint, gold):
        return ""
    if tokenizer is None:
        return hint[:4 * token_budget].strip()
    tokens = tokenizer.encode(hint, add_special_tokens=False)
    if len(tokens) <= token_budget:
        return hint
    return tokenizer.decode(tokens[:token_budget], skip_special_tokens=True).strip()


def build_prompt(question: str, hint: str | None = None) -> str:
    """The cold/contexted prompt from Appendix A.1; gold answers never enter it."""
    if not question.strip():
        raise ValueError("question must be nonempty")
    prefix = f"Question: {question.strip()}\n"
    if hint and hint.strip():
        prefix += f"Hint:\n{hint.strip()}\n"
    return prefix + "Let's solve this step by step:"
