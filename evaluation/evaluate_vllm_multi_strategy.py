#!/usr/bin/env python3
"""
vLLM Multi-Strategy Evaluation Script
======================================

Evaluates models using domain-specific multi-strategy prompts from src/prompts.
Compares results with simple CoT prompts.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer

from src.rewards import CombinedRewardFunction
from src.data import ReasoningDataset
from src.prompts import get_prompt_template
from src.data.preprocessing import (
    format_mistral3_prompt,
    format_phi4_prompt,
    is_mistral3_model,
    is_phi4_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMMultiStrategyEvaluator:
    """Evaluator using multi-strategy prompts."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str = None,
        model_id: str = "model",
        gpu_memory_utilization: float = 0.85,
        base_only: bool = False,
        dataset_name: str = "math",
        tensor_parallel_size: int = 1,
        prompt_template: str = "auto",
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.base_only = base_only
        self.dataset_name = dataset_name
        self.tensor_parallel_size = tensor_parallel_size
        self.base_model = base_model

        # Determine prompt template type
        if prompt_template == "auto":
            if is_phi4_model(base_model):
                self.template_type = "phi-chat"
            elif is_mistral3_model(base_model):
                self.template_type = "mistral-chat"
            else:
                self.template_type = "standard"
        else:
            self.template_type = prompt_template

        # Load tokenizer for chat templates
        self.tokenizer = None
        if self.template_type in ["mistral-chat", "phi-chat"]:
            logger.info(f"Loading tokenizer for {self.template_type} template...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )

        # Get fallback prompt template (for standard mode)
        self.prompt_template = get_prompt_template(dataset_name)
        logger.info(f"Using prompt template: {self.template_type}")
        logger.info(f"  Dataset: {dataset_name}")
        if self.template_type == "standard":
            logger.info(f"  Domain: {self.prompt_template.domain}")
            logger.info(f"  Strategies: {self.prompt_template.strategies}")

        logger.info(f"Initializing vLLM for {model_id}...")
        logger.info(f"  Base model: {base_model}")
        if not base_only:
            logger.info(f"  Adapter: {adapter_path}")
        else:
            logger.info(f"  Mode: BASE ONLY (no adapter)")

        # Initialize vLLM
        if base_only:
            self.llm = LLM(
                model=base_model,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=4096,  # Longer for multi-strategy responses
                tensor_parallel_size=tensor_parallel_size,
            )
            self.lora_request = None
        else:
            self.llm = LLM(
                model=base_model,
                enable_lora=True,
                max_lora_rank=64,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=4096,
                tensor_parallel_size=tensor_parallel_size,
            )
            self.lora_request = LoRARequest(
                lora_name=model_id,
                lora_int_id=1,
                lora_path=adapter_path,
            )

        # Reward function for answer extraction
        self.reward_fn = CombinedRewardFunction({})

        # Sampling params - longer max_tokens for multi-strategy
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,  # Longer for detailed reasoning
        )

    def _format_prompt(self, question: str) -> str:
        """Format a single prompt based on template type."""
        if self.template_type == "mistral-chat" and self.tokenizer:
            return format_mistral3_prompt(
                question=question,
                tokenizer=self.tokenizer,
                dataset=self.dataset_name,
                multi_strategy=True,
            )
        elif self.template_type == "phi-chat" and self.tokenizer:
            return format_phi4_prompt(
                question=question,
                tokenizer=self.tokenizer,
                dataset=self.dataset_name,
                multi_strategy=True,
            )
        else:
            # Standard: use domain-specific prompt template
            return self.prompt_template.format_prompt(question)

    def generate_batch(
        self,
        questions: List[str],
        num_traces: int = 2,  # Fewer traces since each is more detailed
    ) -> List[List[str]]:
        """Generate reasoning traces for a batch of questions."""
        # Format prompts using appropriate template
        prompts = [self._format_prompt(q) for q in questions]

        # Repeat prompts for multiple traces
        all_prompts = []
        for _ in range(num_traces):
            all_prompts.extend(prompts)

        # Generate with vLLM
        if self.lora_request:
            outputs = self.llm.generate(
                all_prompts,
                self.sampling_params,
                lora_request=self.lora_request,
            )
        else:
            outputs = self.llm.generate(
                all_prompts,
                self.sampling_params,
            )

        # Organize outputs by question
        traces_per_question = [[] for _ in range(len(questions))]
        for i, output in enumerate(outputs):
            q_idx = i % len(questions)
            traces_per_question[q_idx].append(output.outputs[0].text)

        return traces_per_question

    def evaluate_dataset(
        self,
        dataset,
        batch_size: int = 16,  # Smaller batch for longer sequences
        num_traces: int = 2,
        max_samples: int = None,
    ) -> Dict:
        """Evaluate on dataset and return per-question results."""
        questions = [s["question"] for s in dataset]
        answers = [s["answer"] for s in dataset]

        if max_samples:
            questions = questions[:max_samples]
            answers = answers[:max_samples]

        all_results = []
        correct_count = 0

        # Process in batches
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating {self.model_id}"):
            batch_q = questions[i:i+batch_size]
            batch_a = answers[i:i+batch_size]
            batch_indices = list(range(i, min(i+batch_size, len(questions))))

            # Generate traces
            batch_traces = self.generate_batch(batch_q, num_traces)

            # Evaluate each question
            for j, (q_idx, question, gt, traces) in enumerate(zip(batch_indices, batch_q, batch_a, batch_traces)):
                is_correct = False
                best_trace = None

                for trace in traces:
                    exploit_result = self.reward_fn.exploit_reward(trace, gt, question)
                    if exploit_result.is_correct:
                        is_correct = True
                        best_trace = trace
                        break

                if is_correct:
                    correct_count += 1

                result = {
                    "question_id": q_idx,
                    "question": question,
                    "ground_truth": gt,
                    "model_id": self.model_id,
                    "is_correct": is_correct,
                    "num_traces": len(traces),
                    "best_trace": best_trace if best_trace else traces[0],
                    "prompt_type": "multi_strategy",
                }
                all_results.append(result)

        total = len(questions)
        metrics = {
            "model_id": self.model_id,
            "total_samples": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total > 0 else 0,
            "prompt_type": "multi_strategy",
            "dataset": self.dataset_name,
        }

        return {"metrics": metrics, "results": all_results}


def main():
    parser = argparse.ArgumentParser(description="vLLM multi-strategy evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory (not needed with --base-only)")
    parser.add_argument("--model", type=str, required=True, choices=["M1", "M2"],
                        help="Which model to evaluate (M1 or M2)")
    parser.add_argument("--base-only", action="store_true",
                        help="Evaluate base model without LoRA adapter")
    parser.add_argument("--dataset", type=str, default="math",
                        help="Dataset name (gsm8k, math, aime, gpqa, medmcqa)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-traces", type=int, default=2,
                        help="Number of traces per question (default: 2 for multi-strategy)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--prompt-template", type=str, default="auto",
                        choices=["auto", "standard", "mistral-chat", "phi-chat"],
                        help="Prompt template type (auto detects from model)")

    args = parser.parse_args()

    # Base model paths
    BASE_MODELS = {
        "M1": "Qwen/Qwen2.5-3B-Instruct",
        "M2": "Qwen/Qwen3-4B-Instruct-2507",
    }

    if args.base_only:
        base_model = BASE_MODELS[args.model]
        adapter_path = None
        model_id = f"{args.model}_base"
        logger.info(f"Evaluating BASE model {args.model}: {base_model}")
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless using --base-only")

        checkpoint_dir = Path(args.checkpoint)
        model_dir = checkpoint_dir / args.model
        adapter_config_path = model_dir / "adapter_config.json"

        if not adapter_config_path.exists():
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

        with open(adapter_config_path) as f:
            config = json.load(f)

        base_model = config.get("base_model_name_or_path")
        adapter_path = str(model_dir)
        model_id = args.model
        logger.info(f"Found {args.model}: base={base_model}")

    # Initialize evaluator with multi-strategy prompts
    evaluator = VLLMMultiStrategyEvaluator(
        base_model=base_model,
        adapter_path=adapter_path,
        model_id=model_id,
        gpu_memory_utilization=args.gpu_memory,
        base_only=args.base_only,
        dataset_name=args.dataset,
        tensor_parallel_size=args.tensor_parallel,
        prompt_template=args.prompt_template,
    )

    # Load dataset
    logger.info(f"Loading {args.dataset} {args.split} split...")
    if args.dataset == "gsm8k":
        dataset = ReasoningDataset.from_gsm8k(split=args.split, shuffle=False)
    elif args.dataset in ["math", "math_qwedsacf"]:
        dataset = ReasoningDataset.from_math(split=args.split, use_qwedsacf=True, shuffle=False)
    elif args.dataset == "aime":
        dataset = ReasoningDataset.from_aime(split=args.split, shuffle=False)
    elif args.dataset in ["gpqa", "gpqa_main", "gpqa_diamond"]:
        # Map dataset name to difficulty parameter
        difficulty_map = {"gpqa": "diamond", "gpqa_main": "main", "gpqa_diamond": "diamond"}
        difficulty = difficulty_map.get(args.dataset, "diamond")
        dataset = ReasoningDataset.from_gpqa(split=args.split, difficulty=difficulty, shuffle=False)
    elif args.dataset == "medmcqa":
        dataset = ReasoningDataset.from_medmcqa(split=args.split, shuffle=False)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Evaluate
    logger.info("Starting multi-strategy evaluation...")
    results = evaluator.evaluate_dataset(
        dataset,
        batch_size=args.batch_size,
        num_traces=args.num_traces,
        max_samples=args.max_samples,
    )

    # Print summary
    metrics = results["metrics"]
    metrics["template_type"] = evaluator.template_type  # Add template info
    logger.info("\n" + "="*50)
    logger.info(f"{args.model} Multi-Strategy Evaluation Results")
    logger.info("="*50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Prompt template: {evaluator.template_type}")
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Correct: {metrics['correct']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")

    # Save results
    output_path = args.output or f"vllm_multi_strategy_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
