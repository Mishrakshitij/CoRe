"""
Data Preprocessing
==================

Utilities for preprocessing reasoning datasets.
"""

import re
from typing import Dict, List, Tuple, Optional


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from \\boxed{...} handling nested braces correctly.

    Examples:
        - \\boxed{42} -> "42"
        - \\boxed{\\frac{1}{2}} -> "\\frac{1}{2}"
        - \\boxed{x^{2}+1} -> "x^{2}+1"

    Args:
        text: Text containing \\boxed{...}

    Returns:
        Content inside boxed, or None if not found
    """
    # Find the start of \boxed{
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None

    start_idx = match.end()  # Position right after the opening {
    brace_count = 1
    idx = start_idx

    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        # idx is now one past the closing brace
        return text[start_idx:idx-1]

    return None


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
    - "\\boxed{X}" (with nested braces support)
    - "#### X"
    - "the answer is X"
    - Last number in the text
    """
    text_lower = text.lower()

    # First try boxed with nested braces support
    boxed_answer = extract_boxed_content(text)
    if boxed_answer:
        return boxed_answer.strip()

    # Then try other patterns
    patterns = [
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


def format_prompt(question: str, few_shot: List[Dict] = None, multi_strategy: bool = False) -> str:
    """
    Format a question into a prompt for the model.

    Args:
        question: The question to solve
        few_shot: Optional few-shot examples
        multi_strategy: If True, use multi-strategy exploration format

    Returns:
        Formatted prompt
    """
    if multi_strategy:
        return format_multi_strategy_prompt(question)

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


# Multi-strategy prompt template for GSM8K
GSM8K_MULTI_STRATEGY_PROMPT = """You are an expert mathematical problem solver. For grade-school math problems, explore multiple distinct solution strategies before arriving at your final answer.

IMPORTANT: Show your work clearly and provide a numerical final answer.

Format your response as:
<strategy id="1">
<approach>Brief name of approach (e.g., "Work Backwards", "Unit Rate", "Algebra")</approach>
<reasoning>
Step-by-step solution using this approach
</reasoning>
<result>
Numerical answer from this approach
</result>
</strategy>

<strategy id="2">
<approach>Alternative approach name</approach>
<reasoning>
Step-by-step solution using the alternative approach
</reasoning>
<result>
Numerical answer from this approach
</result>
</strategy>

<final_answer>
Your final numerical answer
</final_answer>

Question: {question}

Solve using at least 2 different approaches:"""


def format_multi_strategy_prompt(question: str, dataset: str = "gsm8k") -> str:
    """
    Format a question using multi-strategy exploration prompt.

    Args:
        question: The question to solve
        dataset: Dataset name for domain-specific prompts (default: "gsm8k")
                 Supported: "gsm8k", "math", "aime"

    Returns:
        Multi-strategy formatted prompt
    """
    # For backward compatibility, use inline GSM8K template if no dataset specified
    # or if prompts module not available
    if dataset.lower() == "gsm8k":
        return GSM8K_MULTI_STRATEGY_PROMPT.format(question=question)

    # Use domain-specific prompts from prompts module
    try:
        from src.prompts import get_prompt_template
        template = get_prompt_template(dataset)
        return template.format_prompt(question)
    except (ImportError, ValueError):
        # Fallback to GSM8K if prompts module unavailable
        return GSM8K_MULTI_STRATEGY_PROMPT.format(question=question)


def format_multi_strategy_contexted_prompt(
    question: str,
    teacher_context: str,
    dataset: str = "gsm8k"
) -> str:
    """
    Format a contexted prompt with multi-strategy format and teacher hint.

    Args:
        question: The question to solve
        teacher_context: Compressed hint from successful peer model
        dataset: Dataset name for domain-specific prompts (default: "gsm8k")
                 Supported: "gsm8k", "math", "aime"

    Returns:
        Multi-strategy contexted prompt
    """
    # For backward compatibility with GSM8K
    if dataset.lower() == "gsm8k":
        return f"""You are an expert mathematical problem solver. A peer model provided this helpful approach:

<peer_hint>
{teacher_context}
</peer_hint>

Now solve the problem using the hint AND explore an alternative strategy:

Format your response as:
<strategy id="1">
<approach>Approach inspired by peer hint</approach>
<reasoning>
Step-by-step solution using the hint
</reasoning>
<result>
Numerical answer from this approach
</result>
</strategy>

<strategy id="2">
<approach>Your own alternative approach</approach>
<reasoning>
Step-by-step solution using a different method
</reasoning>
<result>
Numerical answer from this approach
</result>
</strategy>

<final_answer>
Your final numerical answer
</final_answer>

Question: {question}

Solve using the hint and an alternative approach:"""

    # Use domain-specific prompts from prompts module
    try:
        from src.prompts import get_prompt_template
        template = get_prompt_template(dataset)
        return template.format_contexted_prompt(question, teacher_context)
    except (ImportError, ValueError):
        # Fallback to GSM8K format
        return format_multi_strategy_contexted_prompt(question, teacher_context, "gsm8k")


def extract_xml_answer(text: str) -> Optional[str]:
    """
    Extract final answer from XML-formatted multi-strategy response.

    Args:
        text: Model response text

    Returns:
        Extracted answer or None
    """
    # Try to extract from <final_answer> tag
    final_match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', text, re.DOTALL)
    if final_match:
        answer = final_match.group(1).strip()
        # Extract number from the answer
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[-1]
        return answer

    # Fallback: try to get from last <result> tag
    results = re.findall(r'<result>\s*(.*?)\s*</result>', text, re.DOTALL)
    if results:
        answer = results[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[-1]
        return answer

    # Final fallback to standard extraction
    return extract_answer(text)


def extract_strategy_blocks(text: str) -> List[Dict]:
    """
    Extract all strategy blocks from multi-strategy response.

    Args:
        text: Model response text

    Returns:
        List of dicts with 'approach', 'reasoning', 'result' keys
    """
    strategies = []

    # Find all strategy blocks
    pattern = r'<strategy\s+id="(\d+)">\s*' \
              r'<approach>(.*?)</approach>\s*' \
              r'<reasoning>(.*?)</reasoning>\s*' \
              r'<result>\s*(.*?)\s*</result>\s*' \
              r'</strategy>'

    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        strategy_id, approach, reasoning, result = match
        strategies.append({
            'id': int(strategy_id),
            'approach': approach.strip(),
            'reasoning': reasoning.strip(),
            'result': result.strip(),
        })

    return strategies


def is_mistral3_model(model_name: str) -> bool:
    """Check if model is a Mistral-3 reasoning model."""
    mistral3_patterns = ["Ministral-3", "ministral-3", "Mistral-3", "mistral-3"]
    return any(pattern in model_name for pattern in mistral3_patterns)


def extract_think_content(text: str) -> Optional[str]:
    """
    Extract content from [THINK]...[/THINK] blocks in Mistral reasoning output.

    Args:
        text: Model response text

    Returns:
        Content between [THINK] and [/THINK] tags, or None if not found
    """
    # Pattern to match [THINK]...[/THINK] with optional whitespace
    pattern = r'\[THINK\](.*?)\[/THINK\]'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def get_text_after_think(text: str) -> str:
    """
    Get text content after [/THINK] tag for answer extraction.

    For Mistral reasoning models, the actual answer comes after the thinking block.

    Args:
        text: Model response text

    Returns:
        Text after [/THINK] or original text if no think block found
    """
    pattern = r'\[/THINK\](.*)$'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return text


# GPQA multi-strategy prompt with XML format for Mistral chat template
GPQA_MISTRAL_XML_FORMAT = """Format your response as:
<strategy id="1">
<approach>Your approach name</approach>
<reasoning>Your detailed reasoning</reasoning>
<result>Letter answer (A, B, C, or D)</result>
</strategy>

<final_answer>Your final letter answer (A, B, C, or D)</final_answer>"""

# AIME multi-strategy prompt with XML format for Mistral chat template
AIME_MISTRAL_XML_FORMAT = """Format your response as:
<strategy id="1">
<approach>Your approach name (e.g., Algebraic, Combinatorial, Geometric)</approach>
<reasoning>Your detailed step-by-step reasoning</reasoning>
<result>Integer answer (000-999)</result>
</strategy>

<final_answer>Your final integer answer (000-999)</final_answer>

IMPORTANT: AIME answers are always integers from 000 to 999."""


def format_mistral3_prompt(
    question: str,
    tokenizer,
    dataset: str = "gpqa",
    multi_strategy: bool = True,
) -> str:
    """
    Format prompt for Mistral-3 reasoning models using chat template.

    Mistral-3 reasoning models expect:
    - System message about thinking process (triggers [THINK] generation)
    - User message with the question and XML format request
    - Model generates [THINK]...[/THINK] followed by XML-formatted response

    Answer format is auto-detected from dataset:
    - GPQA/MedMCQA: MCQ (A/B/C/D)
    - AIME: Integer (000-999)
    - GSM8K/MATH: Numerical

    Args:
        question: The question to solve
        tokenizer: Mistral tokenizer with chat template
        dataset: Dataset type (determines system message and answer format)
        multi_strategy: If True, request multi-strategy XML format

    Returns:
        Formatted prompt using chat template
    """
    # Auto-detect answer format from dataset
    dataset_lower = dataset.lower()

    # Determine if this is a numeric/integer answer dataset
    is_numeric_dataset = dataset_lower in [
        "aime", "gsm8k", "math", "math_qwedsacf",
        "aime-1983-2024", "aime-1983-2025"
    ]
    is_mcq_dataset = dataset_lower in [
        "gpqa", "gpqa_main", "gpqa_diamond", "gpqa_extended",
        "medmcqa"
    ]

    # System message for Mistral reasoning - triggers [THINK] format
    if dataset_lower in ["aime", "aime-1983-2024", "aime-1983-2025"]:
        # AIME/Olympiad-specific system message
        system_message = """You are an expert Olympiad mathematician solving AIME competition problems.

Think deeply using competition-level mathematical reasoning. Consider:
- Algebraic manipulation and clever substitutions
- Combinatorial counting with inclusion-exclusion
- Geometric insights and coordinate geometry
- Number theory and modular arithmetic
- Multiple approaches to verify your answer

AIME answers are always integers from 000 to 999."""
    elif dataset_lower in ["gsm8k"]:
        system_message = """You are an expert mathematician solving grade-school math problems.

Think through the problem step by step. Consider:
- Multiple solution approaches
- Careful arithmetic
- Verification of your answer"""
    elif dataset_lower in ["math", "math_qwedsacf"]:
        system_message = """You are an expert mathematician solving competition-level math problems.

Think through the problem step by step. Consider:
- Algebraic and analytical approaches
- Geometric insights
- Multiple solution strategies to verify your answer"""
    elif dataset_lower in ["gpqa", "gpqa_main", "gpqa_diamond", "gpqa_extended"]:
        system_message = """You are an expert scientist solving graduate-level science questions.

Think through the problem step by step using your reasoning capabilities. Consider:
- Relevant scientific principles and equations
- Multiple approaches to verify your answer
- Process of elimination for multiple choice"""
    elif dataset_lower in ["medmcqa"]:
        system_message = """You are an expert medical professional solving clinical questions.

Think through the problem step by step. Consider:
- Relevant medical knowledge and clinical reasoning
- Process of elimination for multiple choice
- Multiple approaches to verify your answer"""
    else:
        system_message = """You are an expert problem solver.

Think through the problem step by step. Show your reasoning clearly."""

    # Build user message with question and optional XML format
    if multi_strategy:
        # Auto-select XML format based on dataset type
        if is_numeric_dataset:
            xml_format = AIME_MISTRAL_XML_FORMAT
        else:
            xml_format = GPQA_MISTRAL_XML_FORMAT
        user_message = f"""{question}

{xml_format}"""
    else:
        user_message = question

    # Build messages for chat template
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Use tokenizer's chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        # Fallback to simple format if chat template fails
        return f"{system_message}\n\n{user_message}\n\nLet me think through this step by step:"
