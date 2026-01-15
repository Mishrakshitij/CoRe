#!/usr/bin/env python3
"""
Combine vLLM Single-Model Results
=================================

Combines M1 and M2 evaluation results to compute collaboration metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_results(m1_path: str, m2_path: str, output_path: str):
    """Combine M1 and M2 results and compute collaboration metrics."""

    # Load results
    with open(m1_path) as f:
        m1_data = json.load(f)
    with open(m2_path) as f:
        m2_data = json.load(f)

    # FIX: Match by question TEXT instead of question_id
    m1_results = {r["question"]: r for r in m1_data["results"]}
    m2_results = {r["question"]: r for r in m2_data["results"]}

    # Verify same questions
    if set(m1_results.keys()) != set(m2_results.keys()):
        logger.warning("Questions don't match between M1 and M2!")
        common_questions = set(m1_results.keys()) & set(m2_results.keys())
        logger.info(f"Using {len(common_questions)} common questions")
    else:
        common_questions = set(m1_results.keys())

    # Compute combined metrics
    m1_correct = 0
    m2_correct = 0
    combined_correct = 0
    rescue_opportunities = 0
    rescue_successes = 0

    combined_results = []

    # Sort questions for deterministic output
    for q_id, question in enumerate(sorted(common_questions)):
        m1_r = m1_results[question]
        m2_r = m2_results[question]

        m1_is_correct = m1_r["is_correct"]
        m2_is_correct = m2_r["is_correct"]

        if m1_is_correct:
            m1_correct += 1
        if m2_is_correct:
            m2_correct += 1

        any_correct = m1_is_correct or m2_is_correct
        if any_correct:
            combined_correct += 1

        # Rescue: one model saved the other
        if m1_is_correct != m2_is_correct:
            rescue_opportunities += 1
            if any_correct:
                rescue_successes += 1

        combined_results.append({
            "question_id": q_id,
            "question": question,
            "ground_truth": m1_r["ground_truth"],
            "M1_correct": m1_is_correct,
            "M2_correct": m2_is_correct,
            "combined_correct": any_correct,
            "M1_trace": m1_r.get("best_trace", ""),
            "M2_trace": m2_r.get("best_trace", ""),
        })

    total = len(common_questions)
    m1_acc = m1_correct / total if total > 0 else 0
    m2_acc = m2_correct / total if total > 0 else 0
    combined_acc = combined_correct / total if total > 0 else 0
    best_single = max(m1_acc, m2_acc)
    collab_gain = combined_acc - best_single
    rescue_rate = rescue_successes / rescue_opportunities if rescue_opportunities > 0 else 0

    metrics = {
        "total_samples": total,
        "model_accuracies": {
            "M1": m1_acc,
            "M2": m2_acc,
        },
        "M1_correct": m1_correct,
        "M2_correct": m2_correct,
        "best_single_accuracy": best_single,
        "combined_correct": combined_correct,
        "combined_accuracy": combined_acc,
        "collaboration_gain": collab_gain,
        "rescue_opportunities": rescue_opportunities,
        "rescue_successes": rescue_successes,
        "rescue_rate": rescue_rate,
    }

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Combined Evaluation Results")
    logger.info("="*50)
    logger.info(f"Total samples: {total}")
    logger.info(f"\nPer-model accuracy:")
    logger.info(f"  M1: {m1_acc:.2%} ({m1_correct}/{total})")
    logger.info(f"  M2: {m2_acc:.2%} ({m2_correct}/{total})")
    logger.info(f"\nBest single model: {best_single:.2%}")
    logger.info(f"Combined accuracy: {combined_acc:.2%} ({combined_correct}/{total})")
    logger.info(f"Collaboration gain: {collab_gain:+.2%}")
    logger.info(f"Rescue rate: {rescue_rate:.2%} ({rescue_successes}/{rescue_opportunities} opportunities)")

    # Save combined results
    output = {
        "metrics": metrics,
        "results": combined_results,
        "source_files": {
            "M1": m1_path,
            "M2": m2_path,
        }
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nCombined results saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Combine vLLM evaluation results")
    parser.add_argument("--m1", type=str, required=True, help="Path to M1 results JSON")
    parser.add_argument("--m2", type=str, required=True, help="Path to M2 results JSON")
    parser.add_argument("--output", type=str, default="vllm_eval_combined.json",
                        help="Output path for combined results")

    args = parser.parse_args()
    combine_results(args.m1, args.m2, args.output)


if __name__ == "__main__":
    main()
