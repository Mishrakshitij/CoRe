"""
Data Preprocessing
==================

Utilities for preprocessing reasoning datasets.
"""

import re
from typing import Dict, List, Tuple, Optional


def preprocess_gsm8k(item: Dict) -> Dict:
    """
    Preprocess a GSM8K example.

    Args:
        item: Raw GSM8K item with 'question' and 'answer' fields

    Returns:
        Preprocessed item
    """
    question = item["question"].strip()
    full_answer = item["answer"]

    # Extract numerical answer after ####
    if "####" in full_answer:
        parts = full_answer.split("####")
        solution = parts[0].strip()
        answer = parts[1].strip()
    else:
        solution = full_answer
        # Try to extract last number
        numbers = re.findall(r'-?\d+\.?\d*', full_answer)
        answer = numbers[-1] if numbers else full_answer.strip()

    # Clean answer (remove commas, dollar signs, etc.)
    answer = re.sub(r'[$,]', '', answer)

    return {
        "question": question,
        "answer": answer,
        "solution": solution,
    }


def preprocess_math(item: Dict) -> Dict:
    """
    Preprocess a MATH dataset example.

    Args:
        item: Raw MATH item with 'problem' and 'solution' fields

    Returns:
        Preprocessed item
    """
    question = item["problem"].strip()
    solution = item["solution"].strip()

    # Extract answer from \boxed{} if present
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
    if boxed_match:
        answer = boxed_match.group(1)
    else:
        # Try common answer patterns
        patterns = [
            r'(?:the answer is|answer:)\s*([^\n.]+)',
            r'(?:therefore|thus)[,\s]+([^\n.]+)',
        ]
        answer = None
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break

        if answer is None:
            # Last resort: take last line
            answer = solution.split('\n')[-1].strip()

    return {
        "question": question,
        "answer": answer,
        "solution": solution,
        "level": item.get("level", ""),
        "type": item.get("type", ""),
    }


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a reasoning trace.

    Looks for common patterns:
    - "the answer is X"
    - "#### X"
    - "\\boxed{X}"
    - Last number in the text
    """
    text_lower = text.lower()

    # Pattern priority order
    patterns = [
        (r'\\boxed\{([^}]+)\}', text),  # LaTeX boxed
        (r'####\s*([^\n]+)', text),  # GSM8K format
        (r'(?:the answer is|final answer:?)\s*[:\s]*([^\n]+)', text_lower),
        (r'(?:therefore|thus|so|hence)[,\s]+(?:the )?(?:answer is )?([^\n.]+)', text_lower),
    ]

    for pattern, source in patterns:
        match = re.search(pattern, source, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up
            answer = re.sub(r'[,\s]+$', '', answer)
            return answer

    # Fallback: find last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    - Removes currency symbols
    - Removes commas
    - Handles fractions
    - Lowercases
    """
    if answer is None:
        return ""

    answer = str(answer).lower().strip()

    # Remove currency and percent
    answer = re.sub(r'^[\$£€]', '', answer)
    answer = re.sub(r'[%]$', '', answer)

    # Remove thousand separators
    answer = answer.replace(',', '')

    # Handle fractions
    if '/' in answer and not '\\' in answer:  # Avoid LaTeX fractions
        try:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    answer = str(num / denom)
        except ValueError:
            pass

    return answer.strip()


def compare_answers(pred: str, gold: str, tolerance: float = 1e-5) -> bool:
    """
    Compare predicted and gold answers.

    Args:
        pred: Predicted answer
        gold: Gold answer
        tolerance: Numerical tolerance for float comparison

    Returns:
        Whether answers match
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact string match
    if pred_norm == gold_norm:
        return True

    # Numerical comparison
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)

        if gold_num == 0:
            return abs(pred_num) < tolerance
        else:
            return abs(pred_num - gold_num) / abs(gold_num) < tolerance
    except ValueError:
        pass

    return False


def format_prompt(question: str, few_shot: List[Dict] = None) -> str:
    """
    Format a question into a prompt for the model.

    Args:
        question: The question to solve
        few_shot: Optional few-shot examples

    Returns:
        Formatted prompt
    """
    prompt_parts = []

    # System instruction
    prompt_parts.append(
        "Solve the following math problem step by step. "
        "Show your reasoning clearly and provide the final answer."
    )

    # Few-shot examples
    if few_shot:
        prompt_parts.append("\nHere are some examples:\n")
        for i, example in enumerate(few_shot, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"Solution: {example['solution']}")
            prompt_parts.append(f"Answer: {example['answer']}\n")

    # Current question
    prompt_parts.append(f"\nNow solve this problem:")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("\nLet's solve this step by step:")

    return "\n".join(prompt_parts)
