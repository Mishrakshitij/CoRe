#!/usr/bin/env python3
"""
Main Training Script for Collaborative Reasoning
=================================================

Usage:
    python scripts/train.py --config configs/base_config.yaml
    python scripts/train.py --config configs/base_config.yaml --experiment configs/experiments/pairwise_qwen_llama.yaml
    python scripts/train.py --config configs/base_config.yaml --models M1=Qwen/Qwen2.5-3B-Instruct M2=meta-llama/Llama-3.2-3B-Instruct
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import wandb
from src.trainers import CollaborativeTrainer
from src.data import create_dataloaders
from src.utils import setup_logging, apply_profile

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override config into base config."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def parse_model_args(model_args: list) -> list:
    """Parse model arguments in format M1=name M2=name."""
    models = []
    for arg in model_args:
        if "=" in arg:
            model_id, model_name = arg.split("=", 1)
            models.append({"name": model_name, "id": model_id})
    return models


def setup_wandb(config: dict, experiment_name: str):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=config["project"]["name"],
        name=experiment_name,
        config=config,
        tags=["collaborative-reasoning", config["policy_optimization"]["algorithm"]],
    )


def main():
    parser = argparse.ArgumentParser(description="Train collaborative reasoning models")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Path to experiment-specific configuration",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile to apply (overrides config 'profile')",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model specifications (e.g., M1=model_name M2=model_name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=[
            "gsm8k",
            "math",
            "math_qwedsacf",
            "aime",
            "gpqa",
            "gpqa_diamond",
            "gpqa_extended",
            "gpqa_main",
            "medmcqa",
            "combined",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["grpo", "gspo", "sapo", "gspo_sapo_hybrid"],
        help="Policy optimization algorithm",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load base config
    logger.info(f"Loading base config from {args.config}")
    config = load_config(args.config)

    # Merge experiment config if provided
    if args.experiment:
        logger.info(f"Loading experiment config from {args.experiment}")
        exp_config = load_config(args.experiment)
        config = merge_configs(config, exp_config)

    # Apply profile override if present
    config = apply_profile(config, args.profile)

    # Override with command line arguments
    if args.algorithm:
        config["policy_optimization"]["algorithm"] = args.algorithm
    if args.dataset:
        config["training"]["dataset"] = args.dataset
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate

    config["project"]["seed"] = args.seed
    config["use_wandb"] = args.use_wandb

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Parse models
    if args.models:
        model_configs = parse_model_args(args.models)
    elif "models" in config and "m1" in config["models"]:
        # From experiment config
        model_configs = [
            config["models"]["m1"],
            config["models"]["m2"],
        ]
        if "m3" in config["models"]:
            model_configs.append(config["models"]["m3"])
        if "m4" in config["models"]:
            model_configs.append(config["models"]["m4"])
    else:
        active_models = config.get("models", {}).get("active")
        if isinstance(active_models, str):
            active_models = [active_models]
        if active_models:
            available = config["models"]["available"]
            missing = [mid for mid in active_models if mid not in available]
            if missing:
                raise ValueError(f"Unknown model id(s) in models.active: {missing}")
            model_configs = [{"name": available[mid]["name"]} for mid in active_models]
        else:
            # Default: use first two available models
            available = config["models"]["available"]
            model_names = list(available.keys())[:2]
            model_configs = [
                {"name": available[m]["name"]} for m in model_names
            ]

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get("experiment", {}).get("name", "collab")
        output_dir = f"outputs/{exp_name}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging (train_logs/<experiment_name>)
    experiment_name = Path(output_dir).name
    log_root = Path(config.get("project", {}).get("log_dir", "./train_logs"))
    log_dir = log_root / experiment_name
    setup_logging(log_dir=str(log_dir), name="")

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log directory: {log_dir}")
    if config.get("profile"):
        logger.info(f"Profile: {config['profile']}")
    logger.info(f"Models: {[m['name'] for m in model_configs]}")

    # Save config
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Setup wandb
    if args.use_wandb:
        experiment_name = config.get("experiment", {}).get("name", "collab_reasoning")
        setup_wandb(config, experiment_name)

    # Create dataloaders
    dataset_name = config["training"]["dataset"]
    logger.info(f"Loading dataset: {dataset_name}")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, dataset_name=dataset_name
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CollaborativeTrainer(
        config=config,
        model_configs=model_configs,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
    )

    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")

    # Log to wandb
    if args.use_wandb:
        wandb.log({"test": test_metrics})
        wandb.finish()

    logger.info("Training complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
