#!/usr/bin/env python3
"""
Run All Experiments
===================

Launches all experiment configurations for collaborative reasoning.

Experiments:
1. Baselines (single model)
2. Pairwise collaborations (6 pairs)
3. Multi-model collaborations (trios and quad)
4. Ablations
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from itertools import combinations, permutations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "qwen2.5_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "ministral_3b": "ministral/Ministral-3b-instruct",
}

# Algorithm variants
ALGORITHMS = ["grpo", "gspo", "sapo", "gspo_sapo_hybrid"]


def run_experiment(
    name: str,
    models: list,
    algorithm: str,
    extra_args: list = None,
    dry_run: bool = False,
):
    """Run a single experiment."""
    cmd = [
        "python", "scripts/train.py",
        "--config", "configs/base_config.yaml",
        "--algorithm", algorithm,
    ]

    # Add model specifications
    for i, model in enumerate(models, 1):
        cmd.extend(["--models", f"M{i}={model}"])

    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)

    # Add output directory
    output_dir = f"outputs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd.extend(["--output-dir", output_dir])

    logger.info(f"Running experiment: {name}")
    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info("Dry run - not executing")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Experiment failed: {name}")
        logger.error(result.stderr)
    else:
        logger.info(f"Experiment completed: {name}")

    return result


def run_baselines(dry_run: bool = False):
    """Run single-model baselines."""
    logger.info("\n" + "="*50)
    logger.info("Running Baselines")
    logger.info("="*50)

    results = []

    for model_name, model_path in MODELS.items():
        for algo in ALGORITHMS:
            name = f"baseline_{model_name}_{algo}"
            result = run_experiment(
                name=name,
                models=[model_path],
                algorithm=algo,
                extra_args=["--num-epochs", "2"],
                dry_run=dry_run,
            )
            results.append({"name": name, "result": result})

    return results


def run_pairwise(dry_run: bool = False):
    """Run all pairwise collaborations."""
    logger.info("\n" + "="*50)
    logger.info("Running Pairwise Experiments")
    logger.info("="*50)

    results = []
    model_items = list(MODELS.items())

    # All unordered pairs
    for (name1, path1), (name2, path2) in combinations(model_items, 2):
        for algo in ["gspo_sapo_hybrid"]:  # Main algorithm
            exp_name = f"pair_{name1}_{name2}_{algo}"
            result = run_experiment(
                name=exp_name,
                models=[path1, path2],
                algorithm=algo,
                dry_run=dry_run,
            )
            results.append({"name": exp_name, "result": result})

    return results


def run_trios(dry_run: bool = False):
    """Run 3-model collaborations."""
    logger.info("\n" + "="*50)
    logger.info("Running Trio Experiments")
    logger.info("="*50)

    results = []
    model_items = list(MODELS.items())

    # Key trio combinations
    trio_combos = [
        ("qwen2.5_3b", "llama3.2_3b", "ministral_3b"),
        ("qwen3_4b", "llama3.2_3b", "ministral_3b"),
    ]

    for trio in trio_combos:
        paths = [MODELS[m] for m in trio]
        exp_name = f"trio_{'_'.join(trio)}"

        result = run_experiment(
            name=exp_name,
            models=paths,
            algorithm="gspo_sapo_hybrid",
            dry_run=dry_run,
        )
        results.append({"name": exp_name, "result": result})

    return results


def run_quad(dry_run: bool = False):
    """Run 4-model collaboration."""
    logger.info("\n" + "="*50)
    logger.info("Running Quad Experiment")
    logger.info("="*50)

    paths = list(MODELS.values())
    exp_name = "quad_all_models"

    result = run_experiment(
        name=exp_name,
        models=paths,
        algorithm="gspo_sapo_hybrid",
        extra_args=["--batch-size", "2"],  # Lower batch for 4 models
        dry_run=dry_run,
    )

    return [{"name": exp_name, "result": result}]


def run_ablations(dry_run: bool = False):
    """Run ablation experiments."""
    logger.info("\n" + "="*50)
    logger.info("Running Ablation Experiments")
    logger.info("="*50)

    results = []

    # Base pair for ablations
    base_models = [MODELS["qwen2.5_3b"], MODELS["llama3.2_3b"]]

    # Ablation 1: Different algorithms
    for algo in ALGORITHMS:
        exp_name = f"ablation_algo_{algo}"
        result = run_experiment(
            name=exp_name,
            models=base_models,
            algorithm=algo,
            dry_run=dry_run,
        )
        results.append({"name": exp_name, "result": result})

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all collaborative reasoning experiments")

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["all", "baselines", "pairwise", "trios", "quad", "ablations"],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_results.json",
        help="Output file for results summary",
    )

    args = parser.parse_args()

    all_results = []

    experiments_to_run = args.experiments
    if "all" in experiments_to_run:
        experiments_to_run = ["baselines", "pairwise", "trios", "quad", "ablations"]

    if "baselines" in experiments_to_run:
        all_results.extend(run_baselines(args.dry_run))

    if "pairwise" in experiments_to_run:
        all_results.extend(run_pairwise(args.dry_run))

    if "trios" in experiments_to_run:
        all_results.extend(run_trios(args.dry_run))

    if "quad" in experiments_to_run:
        all_results.extend(run_quad(args.dry_run))

    if "ablations" in experiments_to_run:
        all_results.extend(run_ablations(args.dry_run))

    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [r["name"] for r in all_results],
        "total": len(all_results),
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAll experiments complete! Results saved to {args.output}")
    logger.info(f"Total experiments: {len(all_results)}")


if __name__ == "__main__":
    main()
