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
    def from_aime(
        cls,
        split: str = "train",
        max_samples: int = None,
        train_test_split: float = 0.8,
        shuffle: bool = True,
    ) -> "ReasoningDataset":
        """Load AIME dataset from HuggingFace.

        Uses TianHongZXY/aime-1983-2025 dataset (cached locally).
        AIME answers are integers from 000 to 999.

        Args:
            split: "train" or "test"
            max_samples: Maximum samples to load
            train_test_split: Train ratio (default 0.8 = 80:20)
            shuffle: Whether to shuffle data
        """
        logger.info(f"Loading AIME dataset {split} split...")

        # Load from HuggingFace - TianHongZXY/aime-1983-2025 only has "test" split
        full_dataset = load_dataset("TianHongZXY/aime-1983-2025", split="test")

        # Split into train/test (manual split since HF only has "test")
        total_samples = len(full_dataset)
        train_size = int(total_samples * train_test_split)

        if split == "train":
            dataset = full_dataset.select(range(train_size))
        else:  # test
            dataset = full_dataset.select(range(train_size, total_samples))

        logger.info(f"Split: {split}, samples: {len(dataset)} (total: {total_samples})")

        data = []
        for item in dataset:
            # TianHongZXY format: problem, answer fields
            question = item.get("problem", "")
            answer = str(item.get("answer", "")).strip()

            data.append({
                "question": question,
                "answer": answer,
                "solution": "",  # AIME doesn't provide solutions
            })

        return cls(data, max_samples=max_samples, shuffle=shuffle)

    @classmethod
    def from_gpqa(
        cls,
        split: str = "train",
        max_samples: int = None,
        difficulty: str = "diamond",  # "diamond" (hardest), "extended", "main"
        shuffle: bool = True,
    ) -> "ReasoningDataset":
        """Load GPQA (Graduate-Level Google-Proof Q&A) dataset.

        Uses Idavidrein/gpqa dataset - science questions that are hard to Google.
        Multiple choice format with 4 options.

        Args:
            split: "train" or "test" (maps to HF splits)
            max_samples: Maximum samples to load
            difficulty: "diamond" (hardest, 198), "extended" (546), "main" (448)
            shuffle: Whether to shuffle data
        """
        logger.info(f"Loading GPQA {difficulty} dataset {split} split...")

        # Map difficulty to HuggingFace config
        config_map = {
            "diamond": "gpqa_diamond",
            "extended": "gpqa_extended",
            "main": "gpqa_main",
        }
        config = config_map.get(difficulty, "gpqa_diamond")

        # GPQA only has train split, we'll split it ourselves
        # Note: GPQA is a gated dataset - requires HuggingFace authentication
        # Run: huggingface-cli login and request access at https://huggingface.co/datasets/Idavidrein/gpqa
        try:
            full_dataset = load_dataset("Idavidrein/gpqa", config, split="train", trust_remote_code=True)
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "authenticated" in error_msg.lower():
                raise RuntimeError(
                    f"GPQA is a gated dataset. To access it:\n"
                    f"1. Run: huggingface-cli login\n"
                    f"2. Request access at: https://huggingface.co/datasets/Idavidrein/gpqa\n"
                    f"Original error: {e}"
                )
            raise

        # Split into train/test (80:20)
        total_samples = len(full_dataset)
        train_size = int(total_samples * 0.8)

        if split == "train":
            dataset = full_dataset.select(range(train_size))
        else:  # test
            dataset = full_dataset.select(range(train_size, total_samples))

        logger.info(f"Split: {split}, samples: {len(dataset)} (total: {total_samples})")

        data = []
        for item in dataset:
            question = item.get("Question", item.get("question", ""))

            # Get choices - GPQA has choice columns
            choices = []
            correct_idx = None
            for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
                if key in item and item[key]:
                    if key == "Correct Answer":
                        correct_idx = len(choices)
                    choices.append(item[key])

            # Format as MCQ
            choice_labels = ["A", "B", "C", "D"]
            formatted_choices = "\n".join([
                f"{choice_labels[i]}. {choice}"
                for i, choice in enumerate(choices)
            ])

            full_question = f"{question}\n\nChoices:\n{formatted_choices}"

            # Answer is the letter of correct choice
            answer = choice_labels[correct_idx] if correct_idx is not None else "A"

            data.append({
                "question": full_question,
                "answer": answer,
                "solution": "",
                "choices": choices,
                "correct_idx": correct_idx,
                "subject": item.get("Subdomain", item.get("subdomain", "")),
            })

        return cls(data, max_samples=max_samples, shuffle=shuffle)

    @classmethod
    def from_medmcqa(
        cls,
        split: str = "train",
        max_samples: int = None,
        shuffle: bool = True,
    ) -> "ReasoningDataset":
        """Load MedMCQA (Medical Multiple Choice QA) dataset.

        Uses openlifescienceai/medmcqa dataset - medical entrance exam questions.
        Multiple choice format with 4 options.

        Args:
            split: "train", "validation", or "test"
            max_samples: Maximum samples to load
            shuffle: Whether to shuffle data
        """
        logger.info(f"Loading MedMCQA dataset {split} split...")

        # MedMCQA has train/validation/test splits
        hf_split = split if split in ["train", "validation", "test"] else "train"
        dataset = load_dataset("openlifescienceai/medmcqa", split=hf_split)

        logger.info(f"Loaded {len(dataset)} samples from {hf_split} split")

        data = []
        for item in dataset:
            question = item.get("question", "")

            # Get choices
            choices = [
                item.get("opa", ""),
                item.get("opb", ""),
                item.get("opc", ""),
                item.get("opd", ""),
            ]

            # Format as MCQ
            choice_labels = ["A", "B", "C", "D"]
            formatted_choices = "\n".join([
                f"{choice_labels[i]}. {choice}"
                for i, choice in enumerate(choices) if choice
            ])

            full_question = f"{question}\n\nChoices:\n{formatted_choices}"

            # Answer is index 0-3, map to A-D
            correct_idx = item.get("cop", 0)  # cop = correct option (0-indexed)
            answer = choice_labels[correct_idx] if correct_idx < 4 else "A"

            data.append({
                "question": full_question,
                "answer": answer,
                "solution": item.get("exp", ""),  # explanation if available
                "choices": choices,
                "correct_idx": correct_idx,
                "subject": item.get("subject_name", ""),
                "topic": item.get("topic_name", ""),
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
            max_samples=int(num_samples * 0.1) if num_samples else None,
        )
    elif dataset_name == "math":
        full_dataset = ReasoningDataset.from_math(
            split="train",
            max_samples=num_samples,
        )
        test_dataset = ReasoningDataset.from_math(
            split="test",
            max_samples=int(num_samples * 0.1) if num_samples else None,
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
    elif dataset_name == "aime":
        full_dataset = ReasoningDataset.from_aime(
            split="train",
            max_samples=num_samples,
            train_test_split=0.8,
        )
        test_dataset = ReasoningDataset.from_aime(
            split="test",
            max_samples=int(num_samples * 0.1) if num_samples else None,
            train_test_split=0.8,
        )
    elif dataset_name.startswith("gpqa"):
        # Support gpqa, gpqa_diamond, gpqa_extended, gpqa_main
        difficulty = "diamond"  # default
        if "_" in dataset_name:
            difficulty = dataset_name.split("_")[1]
        full_dataset = ReasoningDataset.from_gpqa(
            split="train",
            max_samples=num_samples,
            difficulty=difficulty,
        )
        test_dataset = ReasoningDataset.from_gpqa(
            split="test",
            max_samples=int(num_samples * 0.1) if num_samples else None,
            difficulty=difficulty,
        )
    elif dataset_name == "medmcqa":
        full_dataset = ReasoningDataset.from_medmcqa(
            split="train",
            max_samples=num_samples,
        )
        # MedMCQA has actual validation split
        test_dataset = ReasoningDataset.from_medmcqa(
            split="validation",
            max_samples=int(num_samples * 0.1) if num_samples else None,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: gsm8k, math, math_qwedsacf, aime, gpqa, gpqa_diamond, gpqa_extended, gpqa_main, medmcqa")

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
