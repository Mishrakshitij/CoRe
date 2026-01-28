"""
Base Prompt Template
====================

Abstract base class for domain-specific multi-strategy prompts.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BasePromptTemplate(ABC):
    """
    Abstract base class for multi-strategy prompt templates.

    Subclasses must implement:
    - format_prompt(): Generate the full prompt for a question
    - format_contexted_prompt(): Generate prompt with peer hint
    - get_answer_instruction(): Describe expected answer format

    Attributes:
        domain: Human-readable domain description
        strategies: List of recommended strategy names
        answer_format: Expected answer type ("numerical", "boxed", "integer", "mcq")
    """

    domain: str = "general"
    strategies: List[str] = []
    answer_format: str = "numerical"

    @abstractmethod
    def format_prompt(self, question: str, **kwargs) -> str:
        """
        Format a question into a multi-strategy prompt.

        Args:
            question: The question to solve
            **kwargs: Additional template parameters

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def format_contexted_prompt(
        self,
        question: str,
        teacher_context: str,
        **kwargs
    ) -> str:
        """
        Format a contexted prompt with peer hint.

        Args:
            question: The question to solve
            teacher_context: Compressed hint from successful peer model
            **kwargs: Additional template parameters

        Returns:
            Formatted contexted prompt string
        """
        pass

    @abstractmethod
    def get_answer_instruction(self) -> str:
        """
        Get instruction for expected answer format.

        Returns:
            Human-readable answer format instruction
        """
        pass

    def get_strategy_block(
        self,
        strategy_id: int = 1,
        strategy_outcome_tag: str = "result",
    ) -> str:
        """
        Get the XML template for a single strategy block.

        Args:
            strategy_id: Strategy number (1, 2, etc.)

        Returns:
            XML template string for a strategy
        """
        return f"""<strategy id="{strategy_id}">
<approach>Name of approach</approach>
<reasoning>
Step-by-step solution using this approach
</reasoning>
<{strategy_outcome_tag}>
Answer from this approach
</{strategy_outcome_tag}>
</strategy>"""

    def get_final_answer_block(self) -> str:
        """
        Get the XML template for final answer.

        Returns:
            XML template for final answer tag
        """
        return """<final_answer>
Your final answer
</final_answer>"""
