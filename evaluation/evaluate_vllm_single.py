#!/usr/bin/env python3
"""
vLLM Single-Model Evaluation Script
====================================

Evaluates a single model (M1 or M2) using vLLM for fast inference.
Results are saved with question IDs for later combination.
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

from src.rewards import CombinedRewardFunction
from src.data import ReasoningDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMSingleModelEvaluator:
    """Evaluator for a single model using vLLM."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        model_id: str,
        gpu_memory_utilization: float = 0.85,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path

        logger.info(f"Initializing vLLM for {model_id}...")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Adapter: {adapter_path}")

        # Initialize vLLM with LoRA
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=2048,
        )

        self.lora_request = LoRARequest(
            lora_name=model_id,
            lora_int_id=1,
            lora_local_path=adapter_path,
        )

        # Reward function for answer extraction
        self.reward_fn = CombinedRewardFunction({})

        # Sampling params
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

    def generate_batch(
        self,
        questions: List[str],
        num_traces: int = 4,
    ) -> List[List[str]]:
        """Generate reasoning traces for a batch of questions."""
        # Prepare prompts
        prompts = [f"Question: {q}\n\nLet's solve this step by step:" for q in questions]

        # Repeat prompts for multiple traces
        all_prompts = []
        for _ in range(num_traces):
            all_prompts.extend(prompts)

        # Generate with vLLM
        outputs = self.llm.generate(
            all_prompts,
            self.sampling_params,
            lora_request=self.lora_request,
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
        batch_size: int = 32,
        num_traces: int = 4,
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
                }
                all_results.append(result)

        total = len(questions)
        metrics = {
            "model_id": self.model_id,
            "total_samples": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total > 0 else 0,
        }

        return {"metrics": metrics, "results": all_results}


def main():
    parser = argparse.ArgumentParser(description="vLLM single-model evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--model", type=str, required=True, choices=["M1", "M2"],
                        help="Which model to evaluate (M1 or M2)")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-traces", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu-memory", type=float, default=0.85)

    args = parser.parse_args()

    # Load adapter config
    checkpoint_dir = Path(args.checkpoint)
    model_dir = checkpoint_dir / args.model
    adapter_config_path = model_dir / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

    with open(adapter_config_path) as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    logger.info(f"Found {args.model}: base={base_model}")

    # Initialize evaluator
    evaluator = VLLMSingleModelEvaluator(
        base_model=base_model,
        adapter_path=str(model_dir),
        model_id=args.model,
        gpu_memory_utilization=args.gpu_memory,
    )

    # Load dataset
    logger.info(f"Loading {args.dataset} {args.split} split...")
    if args.dataset == "gsm8k":
        dataset = ReasoningDataset.from_gsm8k(split=args.split)
    else:
        dataset = ReasoningDataset.from_math(split=args.split)

    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_dataset(
        dataset,
        batch_size=args.batch_size,
        num_traces=args.num_traces,
        max_samples=args.max_samples,
    )

    # Print summary
    metrics = results["metrics"]
    logger.info("\n" + "="*50)
    logger.info(f"{args.model} Evaluation Results")
    logger.info("="*50)
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Correct: {metrics['correct']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")

    # Save results
    output_path = args.output or f"vllm_eval_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
