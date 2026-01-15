"""
Reasoning Dataset
=================

Dataset classes for loading and batching reasoning data.
"""

import json
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging

from .preprocessing import extract_boxed_content

logger = logging.getLogger(__name__)


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks (GSM8K, MATH, etc.).

    Each item contains:
    - question: The problem statement
    - answer: The ground truth answer
    - solution: (optional) Full solution trace
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        max_samples: int = None,
        shuffle: bool = True,
    ):
        self.data = data

        if shuffle:
            random.shuffle(self.data)

        if max_samples and max_samples < len(self.data):
            self.data = self.data[:max_samples]

        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "solution": item.get("solution", ""),
            "idx": idx,
        }

    @classmethod
    def from_gsm8k(
        cls,
        split: str = "train",
        max_samples: int = None,
        shuffle: bool = True,  # Set False for deterministic evaluation
    ) -> "ReasoningDataset":
        """Load GSM8K dataset from HuggingFace."""
        logger.info(f"Loading GSM8K {split} split...")

        dataset = load_dataset("gsm8k", "main", split=split)

        data = []
        for item in dataset:
            # Extract answer from solution
            solution = item["answer"]
            # GSM8K format: "... #### ANSWER"
            if "####" in solution:
                answer = solution.split("####")[-1].strip()
            else:
                answer = solution.split("\n")[-1].strip()

            data.append({
                "question": item["question"],
                "answer": answer,
                "solution": solution,
            })

        return cls(data, max_samples=max_samples, shuffle=shuffle)

    @classmethod
    def from_math(
        cls,
        split: str = "train",
        max_samples: int = None,
        difficulty: str = None,  # "easy", "medium", "hard"
        use_qwedsacf: bool = False,  # Use qwedsacf/competition_math instead
        train_test_split: float = 0.8,  # For qwedsacf dataset (only has train)
        shuffle: bool = True,  # Set False for deterministic evaluation
    ) -> "ReasoningDataset":
        """Load MATH dataset from HuggingFace.

        Args:
            split: "train" or "test"
            max_samples: Maximum samples to load
            difficulty: Filter by difficulty ("easy", "medium", "hard")
            use_qwedsacf: If True, use qwedsacf/competition_math (12.5k, train only)
            train_test_split: Train ratio when using qwedsacf (default 0.8 = 80:20)
        """
        import re

        if use_qwedsacf:
            logger.info(f"Loading qwedsacf/competition_math {split} split (80:20 from train)...")
            # qwedsacf/competition_math only has train split with 12.5k samples
            full_dataset = load_dataset("qwedsacf/competition_math", split="train")

            # Split into train/test (80:20)
            total_samples = len(full_dataset)
            train_size = int(total_samples * train_test_split)

            if split == "train":
                dataset = full_dataset.select(range(train_size))
            else:  # test
                dataset = full_dataset.select(range(train_size, total_samples))

            logger.info(f"Split: {split}, samples: {len(dataset)} (total: {total_samples})")
        else:
            logger.info(f"Loading hendrycks/competition_math {split} split...")
            dataset = load_dataset("hendrycks/competition_math", split=split)

        data = []
        for item in dataset:
            # Extract answer from solution
            solution = item["solution"]

            # MATH format often has \boxed{answer} - use helper for nested braces
            answer = extract_boxed_content(solution)
            if not answer:
                answer = solution.split("\n")[-1].strip()

            # Filter by difficulty if specified
            level = item.get("level", "")
            if difficulty:
                if difficulty == "easy" and "1" not in level and "2" not in level:
                    continue
                elif difficulty == "medium" and "3" not in level:
                    continue
                elif difficulty == "hard" and "4" not in level and "5" not in level:
                    continue

            data.append({
                "question": item["problem"],
                "answer": answer,
                "solution": solution,
                "level": level,
                "type": item.get("type", ""),
            })

        return cls(data, max_samples=max_samples, shuffle=shuffle)

    @classmethod
    def from_json(
        cls,
        path: str,
        max_samples: int = None,
    ) -> "ReasoningDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(data, max_samples=max_samples)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        max_samples: int = None,
    ) -> "ReasoningDataset":
        """Load dataset from JSONL file."""
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))

        return cls(data, max_samples=max_samples)


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """Collate function for DataLoader."""
    return {
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "solution": [item["solution"] for item in batch],
        "idx": [item["idx"] for item in batch],
    }


def create_dataloaders(
    config: dict,
    dataset_name: str = "gsm8k",
    accelerator=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Training configuration
        dataset_name: Name of dataset to load
        accelerator: Optional Accelerator instance for distributed training

    Returns:
        (train_loader, val_loader, test_loader)
    """
    num_samples = config["training"]["num_samples"]
    batch_size = config["training"]["batch_size"]
    train_split = config["training"]["train_split"]
    val_split = config["training"]["val_split"]

    # Load dataset
    if dataset_name == "gsm8k":
        full_dataset = ReasoningDataset.from_gsm8k(
            split="train",
            max_samples=num_samples,
        )
        test_dataset = ReasoningDataset.from_gsm8k(
            split="test",
            max_samples=int(num_samples * 0.1),
        )
    elif dataset_name == "math":
        full_dataset = ReasoningDataset.from_math(
            split="train",
            max_samples=num_samples,
        )
        test_dataset = ReasoningDataset.from_math(
            split="test",
            max_samples=int(num_samples * 0.1),
        )
    elif dataset_name == "math_qwedsacf":
        # Use qwedsacf/competition_math with 80:20 train:test split
        full_dataset = ReasoningDataset.from_math(
            split="train",
            max_samples=num_samples,
            use_qwedsacf=True,
            train_test_split=0.8,
        )
        test_dataset = ReasoningDataset.from_math(
            split="test",
            max_samples=int(num_samples * 0.1) if num_samples and num_samples > 0 else None,
            use_qwedsacf=True,
            train_test_split=0.8,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split train and validation
    total = len(full_dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)

    train_data = full_dataset.data[:train_size]
    val_data = full_dataset.data[train_size:train_size + val_size]

    train_dataset = ReasoningDataset(train_data, shuffle=True)
    val_dataset = ReasoningDataset(val_data, shuffle=False)

    # Create dataloaders with optional distributed sampler
    train_sampler = None
    if accelerator is not None and accelerator.num_processes > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
