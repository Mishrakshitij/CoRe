#!/usr/bin/env python3
"""
Distributed Training Script with Accelerate/DeepSpeed
=====================================================

Multi-GPU training for Collaborative Reasoning models.

Usage:
    # Basic multi-GPU with Accelerate (4 GPUs)
    accelerate launch --num_processes 4 scripts/train_distributed.py \
        --config configs/grpo_full_gsm8k.yaml

    # With DeepSpeed ZeRO Stage 2
    accelerate launch --num_processes 4 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed/ds_config_zero2.json \
        scripts/train_distributed.py --config configs/grpo_full_gsm8k.yaml

    # With Accelerate config file
    accelerate launch --config_file configs/accelerate/default_config.yaml \
        scripts/train_distributed.py --config configs/grpo_full_gsm8k.yaml

    # Single GPU (for testing)
    python scripts/train_distributed.py --config configs/grpo_full_gsm8k.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import logging

from src.trainers import DistributedCollaborativeTrainer
from src.data import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Distributed Collaborative Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (optional, can also use accelerate --use_deepspeed)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of training samples",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=None,
        help="Override generation batch size",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.num_samples:
        config["training"]["num_samples"] = args.num_samples

    if args.gen_batch_size:
        if "fast_training" not in config:
            config["fast_training"] = {}
        config["fast_training"]["gen_batch_size"] = args.gen_batch_size

    config["use_wandb"] = args.use_wandb

    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm = config["policy_optimization"]["algorithm"]
    num_samples = config["training"]["num_samples"]
    experiment_name = f"distributed_{algorithm}_{num_samples}samples_{timestamp}"

    output_dir = Path(config["project"]["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")

    # Model configs
    model_configs = [
        config["models"]["available"]["qwen2_5_3b"],
        config["models"]["available"]["qwen3_4b"],
    ]

    # Create dataloaders (without accelerator first - will be wrapped by trainer)
    logger.info(f"Loading dataset: {config['training']['dataset']}")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        config=config,
        dataset_name=config["training"]["dataset"],
    )

    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Val samples: {len(val_dataloader.dataset)}")
    logger.info(f"Test samples: {len(test_dataloader.dataset)}")

    # Create distributed trainer
    trainer = DistributedCollaborativeTrainer(
        config=config,
        model_configs=model_configs,
        output_dir=str(output_dir),
        deepspeed_config=args.deepspeed_config,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Save config
    if trainer.is_main_process:
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    # Train
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
