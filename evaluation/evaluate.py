#!/usr/bin/env python3
"""
Evaluation Script
=================

Comprehensive evaluation for collaborative reasoning models.

Metrics:
- Accuracy (per model and combined)
- Rescue rate
- Diversity metrics
- Collaboration gain
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from src.rewards import CombinedRewardFunction
from src.data import create_dataloaders, ReasoningDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeEvaluator:
    """Evaluator for collaborative reasoning models."""

    def __init__(
        self,
        config: dict,
        checkpoint_dir: str,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)

        # Load models
        self.models = {}
        self.tokenizers = {}
        self._load_models()

        # Reward function for metrics
        self.reward_fn = CombinedRewardFunction(config.get("rewards", {}))

        # Metrics storage
        self.metrics = defaultdict(list)

    def _load_models(self):
        """Load models from checkpoint (supports LoRA adapters)."""
        from peft import PeftModel
        import json

        # Find model directories
        model_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("M")]

        for model_dir in model_dirs:
            model_id = model_dir.name
            logger.info(f"Loading {model_id} from {model_dir}")

            # Check if this is a LoRA adapter checkpoint
            adapter_config_path = model_dir / "adapter_config.json"

            if adapter_config_path.exists():
                # Load LoRA adapter on top of base model
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)

                base_model_name = adapter_config.get("base_model_name_or_path")
                logger.info(f"  Base model: {base_model_name}")
                logger.info(f"  Loading as LoRA adapter...")

                # Load tokenizer from adapter dir (has chat template)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)

                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

                # Load LoRA adapter
                model = PeftModel.from_pretrained(base_model, model_dir)
                model.eval()
            else:
                # Full model checkpoint
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                model.eval()

            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer

    @torch.no_grad()
    def generate(
        self,
        model_id: str,
        question: str,
        num_traces: int = 1,
    ) -> List[str]:
        """Generate reasoning traces."""
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]

        prompt = f"Question: {question}\n\nLet's solve this step by step:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        traces = []
        for _ in range(num_traces):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            trace = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            traces.append(trace)

        return traces

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        num_traces: int = 4,
    ) -> Dict:
        """Evaluate a single question across all models."""
        results = {
            "question": question,
            "ground_truth": ground_truth,
            "models": {},
            "combined": {},
        }

        any_correct = False
        all_traces = []
        all_correct = []

        for model_id in self.models:
            # Generate traces
            traces = self.generate(model_id, question, num_traces)

            # Evaluate each trace
            model_correct = False
            model_traces = []

            for trace in traces:
                exploit_result = self.reward_fn.exploit_reward(
                    trace, ground_truth, question
                )
                model_traces.append({
                    "trace": trace[:500],  # Truncate for storage
                    "is_correct": exploit_result.is_correct,
                    "extracted_answer": exploit_result.extracted_answer,
                })

                if exploit_result.is_correct:
                    model_correct = True
                    any_correct = True

            all_traces.extend(traces)
            all_correct.append(model_correct)

            results["models"][model_id] = {
                "traces": model_traces,
                "any_correct": model_correct,
                "accuracy": sum(1 for t in model_traces if t["is_correct"]) / len(model_traces),
            }

        # Combined metrics
        results["combined"] = {
            "any_correct": any_correct,
            "num_models_correct": sum(all_correct),
            "collaboration_benefit": any_correct and not all(all_correct),
        }

        return results

    def evaluate_dataset(
        self,
        dataloader,
        max_samples: int = None,
    ) -> Dict:
        """Evaluate on a full dataset."""
        all_results = []
        model_accuracies = defaultdict(list)
        combined_correct = 0
        total = 0
        rescue_opportunities = 0
        rescue_successes = 0

        samples = list(dataloader)
        if max_samples:
            samples = samples[:max_samples]

        for batch in tqdm(samples, desc="Evaluating"):
            for question, ground_truth in zip(batch["question"], batch["answer"]):
                result = self.evaluate_single(question, ground_truth)
                all_results.append(result)

                # Per-model accuracy
                for model_id, model_result in result["models"].items():
                    model_accuracies[model_id].append(model_result["any_correct"])

                # Combined metrics
                if result["combined"]["any_correct"]:
                    combined_correct += 1

                # Rescue metrics
                num_correct = result["combined"]["num_models_correct"]
                if num_correct > 0 and num_correct < len(self.models):
                    rescue_opportunities += 1
                    if result["combined"]["any_correct"]:
                        rescue_successes += 1

                total += 1

        # Compute final metrics
        metrics = {
            "total_samples": total,
            "model_accuracies": {
                mid: np.mean(accs) for mid, accs in model_accuracies.items()
            },
            "best_single_accuracy": max(
                np.mean(accs) for accs in model_accuracies.values()
            ),
            "combined_accuracy": combined_correct / total,
            "collaboration_gain": (combined_correct / total) - max(
                np.mean(accs) for accs in model_accuracies.values()
            ),
            "rescue_rate": rescue_successes / rescue_opportunities if rescue_opportunities > 0 else 0,
            "rescue_opportunities": rescue_opportunities,
        }

        return {
            "metrics": metrics,
            "results": all_results,
        }

    def compute_diversity_metrics(
        self,
        traces: List[str],
    ) -> Dict:
        """Compute diversity metrics for a set of traces."""
        if len(traces) < 2:
            return {"pairwise_distance": 0.0, "unique_strategies": 1}

        # Use explore reward's distance function
        distances = []
        for i, t1 in enumerate(traces):
            for t2 in traces[i+1:]:
                d = self.reward_fn.explore_reward.compute_distance(t1, t2)
                distances.append(d)

        # Extract operation signatures
        signatures = set()
        for trace in traces:
            sig = self.reward_fn.explore_reward.extract_plan_signature(trace)
            signatures.add(sig)

        return {
            "pairwise_distance": np.mean(distances) if distances else 0.0,
            "unique_strategies": len(signatures),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate collaborative reasoning models")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=4,
        help="Number of traces per model per question",
    )

    args = parser.parse_args()

    # Load config from checkpoint
    config_path = Path(args.checkpoint) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Initialize evaluator
    evaluator = CollaborativeEvaluator(config, args.checkpoint)

    # Load dataset
    logger.info(f"Loading {args.dataset} {args.split} split...")
    if args.dataset == "gsm8k":
        dataset = ReasoningDataset.from_gsm8k(split=args.split)
    else:
        dataset = ReasoningDataset.from_math(split=args.split)

    from torch.utils.data import DataLoader
    from src.data.dataset import collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_dataset(dataloader, max_samples=args.max_samples)

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)

    metrics = results["metrics"]
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"\nPer-model accuracy:")
    for model_id, acc in metrics["model_accuracies"].items():
        logger.info(f"  {model_id}: {acc:.2%}")
    logger.info(f"\nBest single model: {metrics['best_single_accuracy']:.2%}")
    logger.info(f"Combined accuracy: {metrics['combined_accuracy']:.2%}")
    logger.info(f"Collaboration gain: {metrics['collaboration_gain']:+.2%}")
    logger.info(f"Rescue rate: {metrics['rescue_rate']:.2%} ({metrics['rescue_opportunities']} opportunities)")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
