"""
Collaborative Reasoning with GRPO/GSPO/SAPO
============================================

A framework for training multiple language models to solve reasoning problems collaboratively.

Modules:
- losses: Policy optimization loss functions (GRPO, GSPO, SAPO, Hybrid)
- rewards: Reward functions (explore, exploit, cross-model)
- trainers: Training loops and collaborative logic
- data: Dataset loading and preprocessing
- utils: Utility functions

Example:
    from src.trainers import CollaborativeTrainer
    from src.data import create_dataloaders

    trainer = CollaborativeTrainer(config, model_configs, output_dir)
    train_loader, val_loader, test_loader = create_dataloaders(config)
    trainer.train(train_loader, val_loader)
"""

__version__ = "0.1.0"
