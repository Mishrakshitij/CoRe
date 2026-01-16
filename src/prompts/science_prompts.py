"""
Science and Medical Dataset Prompts
====================================

Multi-strategy prompt templates for science QA and medical MCQ datasets:
- GPQA: Graduate-level science questions (physics, chemistry, biology)
- MedMCQA: Medical entrance exam questions
"""

from typing import List
from .base import BasePromptTemplate


class GPQAPrompt(BasePromptTemplate):
    """
    Multi-strategy prompt for GPQA (Graduate-Level Google-Proof Q&A).

    GPQA contains hard science questions that require deep domain knowledge.
    Multiple choice format with 4 options (A, B, C, D).

    Strategies focus on scientific reasoning:
    - First Principles: Derive from fundamental laws/principles
    - Process of Elimination: Rule out incorrect options systematically
    - Domain Knowledge: Apply specific scientific concepts
    """

    domain = "graduate-level science"
    strategies = [
        "First Principles",
        "Process of Elimination",
        "Domain Knowledge Application",
        "Dimensional Analysis"
    ]
    answer_format = "mcq"

    TEMPLATE = """You are an expert scientist solving graduate-level science questions. For this multiple choice question, analyze each option carefully using multiple approaches before selecting your final answer.

IMPORTANT: This is a multiple choice question. Your final answer must be a single letter (A, B, C, or D).

Format your response as:
<strategy id="1">
<approach>Name of approach (e.g., "First Principles", "Process of Elimination", "Domain Knowledge")</approach>
<reasoning>
Detailed step-by-step analysis using this approach.
Consider relevant scientific principles, equations, or concepts.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<strategy id="2">
<approach>Alternative approach name</approach>
<reasoning>
Analyze the question using a different scientific perspective or method.
Verify your answer by checking consistency with known principles.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<final_answer>
Your final letter answer (A, B, C, or D)
</final_answer>

{question}

Solve using at least 2 different approaches:"""

    CONTEXTED_TEMPLATE = """You are an expert scientist solving graduate-level science questions. A peer model provided this helpful approach:

<peer_hint>
{teacher_context}
</peer_hint>

Now solve the problem using the hint AND explore an alternative strategy.

IMPORTANT: Your final answer must be a single letter (A, B, C, or D).

Format your response as:
<strategy id="1">
<approach>Approach inspired by peer hint</approach>
<reasoning>
Analysis using the hint's insight and scientific principles.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<strategy id="2">
<approach>Your own alternative approach</approach>
<reasoning>
Independent analysis using different scientific reasoning.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<final_answer>
Your final letter answer (A, B, C, or D)
</final_answer>

{question}

Solve using the hint and an alternative approach:"""

    def format_prompt(self, question: str, **kwargs) -> str:
        return self.TEMPLATE.format(question=question)

    def format_contexted_prompt(
        self,
        question: str,
        teacher_context: str,
        **kwargs
    ) -> str:
        return self.CONTEXTED_TEMPLATE.format(
            question=question,
            teacher_context=teacher_context
        )

    def get_answer_instruction(self) -> str:
        return "Provide a single letter answer (A, B, C, or D)"


class MedMCQAPrompt(BasePromptTemplate):
    """
    Multi-strategy prompt for MedMCQA (Medical Multiple Choice QA).

    MedMCQA contains medical entrance exam questions covering various
    medical subjects. Multiple choice format with 4 options (A, B, C, D).

    Strategies focus on medical reasoning:
    - Clinical Reasoning: Apply clinical knowledge and diagnostic thinking
    - Anatomical/Physiological: Use foundational medical science
    - Pharmacological: Consider drug mechanisms and interactions
    - Process of Elimination: Rule out incorrect options
    """

    domain = "medical science"
    strategies = [
        "Clinical Reasoning",
        "Anatomical/Physiological Basis",
        "Pharmacological Analysis",
        "Process of Elimination"
    ]
    answer_format = "mcq"

    TEMPLATE = """You are an expert medical professional solving medical entrance exam questions. For this multiple choice question, analyze each option carefully using multiple approaches before selecting your final answer.

IMPORTANT: This is a multiple choice question. Your final answer must be a single letter (A, B, C, or D).

Format your response as:
<strategy id="1">
<approach>Name of approach (e.g., "Clinical Reasoning", "Anatomical Basis", "Pharmacological")</approach>
<reasoning>
Detailed medical reasoning using this approach.
Consider relevant pathophysiology, clinical presentations, or mechanisms.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<strategy id="2">
<approach>Alternative approach name</approach>
<reasoning>
Analyze the question using a different medical perspective.
Consider differential diagnoses or alternative mechanisms.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<final_answer>
Your final letter answer (A, B, C, or D)
</final_answer>

{question}

Solve using at least 2 different approaches:"""

    CONTEXTED_TEMPLATE = """You are an expert medical professional solving medical entrance exam questions. A peer model provided this helpful approach:

<peer_hint>
{teacher_context}
</peer_hint>

Now solve the problem using the hint AND explore an alternative strategy.

IMPORTANT: Your final answer must be a single letter (A, B, C, or D).

Format your response as:
<strategy id="1">
<approach>Approach inspired by peer hint</approach>
<reasoning>
Analysis using the hint's medical insight.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<strategy id="2">
<approach>Your own alternative approach</approach>
<reasoning>
Independent analysis using different medical reasoning.
</reasoning>
<result>
Letter answer (A, B, C, or D)
</result>
</strategy>

<final_answer>
Your final letter answer (A, B, C, or D)
</final_answer>

{question}

Solve using the hint and an alternative approach:"""

    def format_prompt(self, question: str, **kwargs) -> str:
        return self.TEMPLATE.format(question=question)

    def format_contexted_prompt(
        self,
        question: str,
        teacher_context: str,
        **kwargs
    ) -> str:
        return self.CONTEXTED_TEMPLATE.format(
            question=question,
            teacher_context=teacher_context
        )

    def get_answer_instruction(self) -> str:
        return "Provide a single letter answer (A, B, C, or D)"
