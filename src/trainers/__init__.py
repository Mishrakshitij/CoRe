"""
Trainers for Collaborative Reasoning
====================================

Implements the main training loops for:
- Single model GRPO/GSPO/SAPO
- Pairwise collaborative training
- Multi-model collaborative training
"""

from .base_trainer import BaseCollabTrainer
from .collab_trainer import CollaborativeTrainer
from .micro_rounds import MicroRoundManager
from .buddy_buffer import BuddyBuffer
from .fast_trainer import FastCollaborativeTrainer

__all__ = [
    "BaseCollabTrainer",
    "CollaborativeTrainer",
    "FastCollaborativeTrainer",
    "MicroRoundManager",
    "BuddyBuffer",
]
