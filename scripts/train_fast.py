#!/usr/bin/env python3
"""
Fast Training Script
====================

Uses batched generation and torch.compile for faster training.
Expected speedup: 3-5x compared to standard training.

Usage:
    python scripts/train_fast.py --config configs/grpo_fast_1000.yaml
    python scripts/train_fast.py --config configs/grpo_fast_1000.yaml --algorithm grpo --num-samples 1000
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
from src.trainers import FastCollaborativeTrainer
from src.data import create_dataloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_fast.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_wandb(config: dict, experiment_name: str):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=config["project"]["name"],
        name=experiment_name,
        config=config,
        tags=["fast-training", config["policy_optimization"]["algorithm"]],
    )


def main():
    parser = argparse.ArgumentParser(description="Fast collaborative training")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_fast_1000.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["grpo", "gspo", "sapo", "hybrid"],
        help="Override algorithm from config",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset from config",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of training samples",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for debugging)",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=4,
        help="Batch size for generation (larger = faster but more memory)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override config with command line args
    if args.algorithm:
        config["policy_optimization"]["algorithm"] = args.algorithm
    if args.dataset:
        config["training"]["dataset"] = args.dataset
    if args.num_samples:
        config["training"]["num_samples"] = args.num_samples

    # Add fast training config
    if "fast_training" not in config:
        config["fast_training"] = {}
    config["fast_training"]["gen_batch_size"] = args.gen_batch_size
    config["fast_training"]["use_compile"] = not args.no_compile

    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm = config["policy_optimization"]["algorithm"]
    num_samples = config["training"]["num_samples"]
    experiment_name = f"fast_{algorithm}_{num_samples}samples_{timestamp}"

    # Setup output directory (use checkpoint parent if resuming)
    if args.resume:
        output_dir = Path(args.resume).parent
        logger.info(f"Resuming from checkpoint: {args.resume}")
    else:
        output_dir = Path(config["project"]["output_dir"]) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Dataset: {config['training']['dataset']}")
    logger.info(f"Num samples: {num_samples}")
    logger.info(f"Generation batch size: {args.gen_batch_size}")
    logger.info(f"torch.compile enabled: {not args.no_compile}")

    # Setup wandb
    if args.use_wandb:
        config["use_wandb"] = True
        setup_wandb(config, experiment_name)

    # Get model configs
    model_configs = [
        config["models"]["available"]["qwen2_5_3b"],
        config["models"]["available"]["qwen3_4b"],
    ]

    logger.info(f"Models:")
    for i, mc in enumerate(model_configs):
        logger.info(f"  M{i+1}: {mc['name']}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        config=config,
        dataset_name=config["training"]["dataset"],
    )

    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    eval_dataloader = val_dataloader if val_dataloader else None
    if eval_dataloader:
        logger.info(f"Eval samples: {len(eval_dataloader.dataset)}")

    # Create fast trainer
    logger.info("Initializing fast trainer...")
    trainer = FastCollaborativeTrainer(
        config=config,
        model_configs=model_configs,
        output_dir=str(output_dir),
        use_compile=not args.no_compile,
    )

    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resuming from epoch {trainer.state.epoch}, step {trainer.state.global_step}")

    # Start training
    logger.info("Starting fast training...")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    logger.info("Training complete!")

    # Cleanup wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
