"""
Domain-Specific Multi-Strategy Prompts
======================================

Provides domain-specific prompt templates for multi-strategy reasoning.

Supported datasets:
- GSM8K: Grade-school math (numerical answers)
- MATH: Competition math (LaTeX boxed answers)
- AIME: AMC/AIME competition (integer 0-999)

Usage:
    from src.prompts import get_prompt_template

    template = get_prompt_template("gsm8k")
    prompt = template.format_prompt(question)
"""

from typing import Dict, Type

from .base import BasePromptTemplate
from .math_prompts import GSM8KPrompt, MATHPrompt, AIMEPrompt


# Registry of prompt templates by dataset name
_PROMPT_REGISTRY: Dict[str, Type[BasePromptTemplate]] = {
    "gsm8k": GSM8KPrompt,
    "math": MATHPrompt,
    "aime": AIMEPrompt,
}

# Aliases for common variations
_PROMPT_ALIASES = {
    "competition_math": "math",
    "qwedsacf/competition_math": "math",
    "aime-1983-2025": "aime",
}


def get_prompt_template(dataset: str) -> BasePromptTemplate:
    """
    Get the appropriate prompt template for a dataset.

    Args:
        dataset: Dataset name (e.g., "gsm8k", "math", "aime")

    Returns:
        Instantiated prompt template

    Raises:
        ValueError: If dataset is not supported
    """
    # Normalize dataset name
    dataset_lower = dataset.lower()

    # Check aliases
    if dataset_lower in _PROMPT_ALIASES:
        dataset_lower = _PROMPT_ALIASES[dataset_lower]

    if dataset_lower not in _PROMPT_REGISTRY:
        available = list(_PROMPT_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {available}"
        )

    return _PROMPT_REGISTRY[dataset_lower]()


def register_prompt_template(name: str, template_class: Type[BasePromptTemplate]):
    """
    Register a new prompt template.

    Args:
        name: Dataset name to register
        template_class: Prompt template class
    """
    _PROMPT_REGISTRY[name.lower()] = template_class


__all__ = [
    "BasePromptTemplate",
    "GSM8KPrompt",
    "MATHPrompt",
    "AIMEPrompt",
    "get_prompt_template",
    "register_prompt_template",
]
