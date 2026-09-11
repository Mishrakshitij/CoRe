"""Conservative answer extraction shared by training, hints and evaluation.

The paper names several output formats but does not specify a symbolic verifier.
This module supports explicit answer markers, nested LaTeX boxes, numerical
equivalence, and exact normalized text. It deliberately never evaluates code or
guesses that the last number in a reasoning trace is its answer.
"""

from __future__ import annotations

from fractions import Fraction
import re
from typing import NamedTuple


class Strategy(NamedTuple):
    identifier: str
    text: str
    answer: str | None


_ANSWER_TAG = re.compile(
    r"<(?P<answer_tag>final[ _-]?answer|answer|outcome|result|strategy_[\w-]+_outcome)\b[^>]*>"
    r"(?P<answer_value>.*?)</(?P=answer_tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_LINE = re.compile(
    r"(?:^|\n)[ \t]*(?:#{4}\s*|(?:\*\*)?(?:final\s+)?answer"
    r"(?:\*\*)?\s*(?::|=|\bis\b)\s*(?:\*\*)?)([^\n]+)",
    re.IGNORECASE,
)
_STRATEGY = re.compile(r"<strategy\b([^>]*)>(.*?)</strategy\s*>", re.I | re.S)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Return balanced ``\\boxed{...}`` spans, including nested braces."""
    result = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, cursor, start = 1, match.end(), match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        if depth == 0:
            result.append((match.start(), cursor, text[start:cursor - 1]))
    return result


def extract_answer(text: str) -> str | None:
    """Extract the latest explicit answer; accept a bare short answer as fallback.

    Plain multi-line reasoning and sentences containing unmarked numbers return
    ``None``. Explicit markers from all supported formats compete by position,
    so a later correction takes precedence over an earlier boxed result.
    """
    candidates: list[tuple[int, str]] = []
    candidates.extend((m.start(), m.group("answer_value")) for m in _ANSWER_TAG.finditer(text))
    candidates.extend((m.start(), m.group(1)) for m in _ANSWER_LINE.finditer(text))
    boxes = boxed_spans(text)
    candidates.extend((start, value) for start, end, value in boxes
                      if not any(other_start < start and other_end >= end
                                 for other_start, other_end, _ in boxes))
    if candidates:
        value = max(candidates, key=lambda pair: pair[0])[1].strip()
        return value or None
    value = text.strip()
    if not value:
        return None
    if _numeric_value(value) is not None or re.fullmatch(r"[\[(]?[A-Z][\])]?[.!]?", value):
        return value
    if re.fullmatch(r"\[[^\[\]\n]+\]", value):
        return value[1:-1].strip() or None
    return None


def _clean(text: str) -> str:
    text = text.strip().replace("−", "-").replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", text)
    text = text.strip("$* \t\r\n")
    text = text.rstrip(".。!")
    if len(text) >= 2 and (text[0], text[-1]) in {("[", "]"), ("(", ")")}:
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def _numeric_value(text: str) -> Fraction | None:
    value = _clean(text)
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    latex_fraction = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", value)
    if latex_fraction:
        value = f"{latex_fraction[1]}/{latex_fraction[2]}"
    parts = re.split(r"\s*/\s*", value)
    if len(parts) not in {1, 2} or not all(_NUMBER.fullmatch(part) for part in parts):
        return None
    try:
        # Limit lengths to avoid pathological huge-number conversions.
        if len(value) > 256 or any(abs(int(e)) > 1000 for e in re.findall(r"[eE]([+-]?\d+)", value)):
            return None
        numerator = Fraction(parts[0])
        return numerator if len(parts) == 1 else numerator / Fraction(parts[1])
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def normalize_answer(text: str | None) -> str:
    """Canonicalize exact numeric values or whitespace/case of other answers.

    Percentages, units and algebraic expressions are retained as text: this is
    intentionally less permissive than a dataset-specific symbolic verifier.
    """
    if text is None:
        return ""
    value = str(text).strip()
    explicit = extract_answer(value)
    if explicit is not None:
        value = explicit
    number = _numeric_value(value)
    if number is not None:
        return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"
    return _clean(value).casefold()


def answers_equal(prediction: str | None, gold: str | None) -> bool:
    """Compare nonempty normalized answers; two missing answers are not correct."""
    prediction, gold = normalize_answer(prediction), normalize_answer(gold)
    return bool(prediction and gold and prediction == gold)


def extract_strategies(text: str) -> list[Strategy]:
    strategies = []
    for index, match in enumerate(_STRATEGY.finditer(text)):
        identifier = re.search(r"\bid\s*=\s*['\"]([^'\"]+)['\"]", match[1], re.I)
        strategies.append(Strategy(identifier[1] if identifier else str(index + 1), match[2].strip(), extract_answer(match[2])))
    return strategies


def select_correct_strategy(text: str, gold: str) -> str | None:
    """Return the first strategy with a verified matching outcome, if available."""
    return next((s.text for s in extract_strategies(text) if answers_equal(s.answer, gold)), None)
