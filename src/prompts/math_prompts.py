"""
Mathematical Dataset Prompts
============================

Multi-strategy prompt templates for mathematical reasoning datasets:
- GSM8K: Grade-school math problems
- MATH: Competition mathematics (AMC, AIME, Olympiad level)
- AIME: American Invitational Mathematics Examination
"""

from typing import List
from .base import BasePromptTemplate


class GSM8KPrompt(BasePromptTemplate):
    """
    Multi-strategy prompt for GSM8K grade-school math problems.

    Strategies focus on arithmetic and basic problem-solving:
    - Work Backwards: Start from the goal
    - Unit Rate: Use rates and proportions
    - Algebra: Set up equations
    """

    domain = "grade-school mathematics"
    strategies = ["Work Backwards", "Unit Rate", "Algebra", "Direct Calculation"]
    answer_format = "numerical"

    TEMPLATE = """You are an expert mathematical problem solver. For grade-school math problems, explore multiple distinct solution strategies before arriving at your final answer.

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

    CONTEXTED_TEMPLATE = """You are an expert mathematical problem solver. A peer model provided this helpful approach:

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
        return "Provide a numerical answer (e.g., 42, $150, 3.5)"


class MATHPrompt(BasePromptTemplate):
    """
    Multi-strategy prompt for competition mathematics (MATH dataset).

    Strategies focus on advanced problem-solving techniques:
    - Direct Algebraic: Manipulate equations directly
    - Pattern Recognition: Identify mathematical patterns or formulas
    - Case Analysis: Break into cases and enumerate

    Answers should be in LaTeX \\boxed{} format.
    """

    domain = "competition mathematics"
    strategies = [
        "Direct Algebraic Manipulation",
        "Pattern Recognition",
        "Case Analysis",
        "Substitution/Transformation"
    ]
    answer_format = "boxed"

    TEMPLATE = """You are an expert competition mathematics solver. For this problem, explore multiple distinct solution strategies before arriving at your final answer.

IMPORTANT: This is a competition math problem. Your final answer should be in LaTeX \\boxed{{}} format.

Format your response as:
<strategy id="1">
<approach>Name of approach (e.g., "Direct Algebra", "Pattern Recognition", "Casework")</approach>
<reasoning>
Detailed step-by-step solution using this approach.
Show all algebraic manipulations clearly.
</reasoning>
<result>
\\boxed{{your answer}}
</result>
</strategy>

<strategy id="2">
<approach>Alternative approach name</approach>
<reasoning>
Detailed step-by-step solution using a different method.
This could use different techniques like substitution, symmetry, or a known theorem.
</reasoning>
<result>
\\boxed{{your answer}}
</result>
</strategy>

<final_answer>
\\boxed{{your final answer}}
</final_answer>

Problem: {question}

Solve using at least 2 different approaches:"""

    CONTEXTED_TEMPLATE = """You are an expert competition mathematics solver. A peer model provided this helpful approach:

<peer_hint>
{teacher_context}
</peer_hint>

Now solve the problem using the hint AND explore an alternative strategy.

IMPORTANT: Your final answer should be in LaTeX \\boxed{{}} format.

Format your response as:
<strategy id="1">
<approach>Approach inspired by peer hint</approach>
<reasoning>
Step-by-step solution using the hint's insight.
</reasoning>
<result>
\\boxed{{your answer}}
</result>
</strategy>

<strategy id="2">
<approach>Your own alternative approach</approach>
<reasoning>
Step-by-step solution using a different method.
</reasoning>
<result>
\\boxed{{your answer}}
</result>
</strategy>

<final_answer>
\\boxed{{your final answer}}
</final_answer>

Problem: {question}

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
        return "Provide your answer in LaTeX \\boxed{} format (e.g., \\boxed{x^2 + 1})"


class AIMEPrompt(BasePromptTemplate):
    """
    Multi-strategy prompt for AIME (American Invitational Mathematics Examination).

    AIME answers are always integers from 000 to 999.

    Strategies focus on competition-style problem solving:
    - Algebraic/Analytic: Direct calculation and manipulation
    - Combinatorial: Counting arguments
    - Geometric/Visual: Coordinate geometry or transformations
    """

    domain = "AIME competition mathematics"
    strategies = [
        "Algebraic/Analytic",
        "Combinatorial/Counting",
        "Geometric/Coordinate",
        "Modular Arithmetic"
    ]
    answer_format = "integer"

    TEMPLATE = """You are an expert AIME competition solver. For this problem, explore multiple distinct solution strategies before arriving at your final answer.

IMPORTANT: AIME answers are always integers from 000 to 999. Your final answer must be a three-digit integer (including leading zeros if needed, e.g., 042).

Format your response as:
<strategy id="1">
<approach>Name of approach (e.g., "Algebraic", "Combinatorial", "Coordinate Geometry")</approach>
<reasoning>
Detailed step-by-step solution using this approach.
For AIME, be careful with:
- Modular arithmetic
- Counting overcounts/undercounts
- Edge cases
</reasoning>
<result>
Integer answer (000-999)
</result>
</strategy>

<strategy id="2">
<approach>Alternative approach name</approach>
<reasoning>
Detailed step-by-step solution using a different method.
Cross-check your count or calculation.
</reasoning>
<result>
Integer answer (000-999)
</result>
</strategy>

<final_answer>
Your final integer answer (000-999)
</final_answer>

Problem: {question}

Solve using at least 2 different approaches:"""

    CONTEXTED_TEMPLATE = """You are an expert AIME competition solver. A peer model provided this helpful approach:

<peer_hint>
{teacher_context}
</peer_hint>

Now solve the problem using the hint AND explore an alternative strategy.

IMPORTANT: AIME answers are always integers from 000 to 999.

Format your response as:
<strategy id="1">
<approach>Approach inspired by peer hint</approach>
<reasoning>
Step-by-step solution using the hint's insight.
</reasoning>
<result>
Integer answer (000-999)
</result>
</strategy>

<strategy id="2">
<approach>Your own alternative approach</approach>
<reasoning>
Step-by-step solution using a different method.
</reasoning>
<result>
Integer answer (000-999)
</result>
</strategy>

<final_answer>
Your final integer answer (000-999)
</final_answer>

Problem: {question}

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
        return "Provide an integer from 000 to 999"
