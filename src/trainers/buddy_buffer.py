"""
Buddy Buffer for Cross-Teaching
===============================

Stores successful teacher contexts from rescue scenarios
for distillation in later epochs.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path


@dataclass
class BuddyEntry:
    """Entry in the buddy buffer."""
    question: str
    teacher_context: str
    rescued_model: str
    source_model: str
    original_trace: str
    rescue_trace: str
    timestamp: int  # Training step when added


class BuddyBuffer:
    """
    Buffer for storing successful rescue contexts.

    Used for distillation training in epoch 2.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.stats = {
            "total_added": 0,
            "rescues_by_model": {},
            "rescues_from_model": {},
        }

    def add(
        self,
        question: str,
        teacher_context: str,
        rescued_model: str,
        source_model: str,
        original_trace: str = "",
        rescue_trace: str = "",
        step: int = 0,
    ):
        """Add a successful rescue to the buffer."""
        entry = BuddyEntry(
            question=question,
            teacher_context=teacher_context,
            rescued_model=rescued_model,
            source_model=source_model,
            original_trace=original_trace,
            rescue_trace=rescue_trace,
            timestamp=step,
        )
        self.buffer.append(entry)

        # Update stats
        self.stats["total_added"] += 1
        self.stats["rescues_by_model"][rescued_model] = \
            self.stats["rescues_by_model"].get(rescued_model, 0) + 1
        self.stats["rescues_from_model"][source_model] = \
            self.stats["rescues_from_model"].get(source_model, 0) + 1

    def sample(
        self,
        batch_size: int,
        model_filter: str = None,
    ) -> List[BuddyEntry]:
        """
        Sample entries from the buffer.

        Args:
            batch_size: Number of entries to sample
            model_filter: Only return entries for this model

        Returns:
            List of BuddyEntry
        """
        if not self.buffer:
            return []

        if model_filter:
            filtered = [e for e in self.buffer if e.rescued_model == model_filter]
        else:
            filtered = list(self.buffer)

        if not filtered:
            return []

        return random.sample(filtered, min(batch_size, len(filtered)))

    def sample_for_distillation(
        self,
        batch_size: int,
    ) -> Dict[str, List[BuddyEntry]]:
        """
        Sample entries grouped by rescued model for distillation.

        Returns:
            Dict mapping model_id to list of entries
        """
        # Group by rescued model
        by_model: Dict[str, List[BuddyEntry]] = {}
        for entry in self.buffer:
            if entry.rescued_model not in by_model:
                by_model[entry.rescued_model] = []
            by_model[entry.rescued_model].append(entry)

        # Sample from each
        result = {}
        per_model = max(1, batch_size // len(by_model)) if by_model else 0

        for model_id, entries in by_model.items():
            result[model_id] = random.sample(entries, min(per_model, len(entries)))

        return result

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            **self.stats,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str):
        """Save buffer to disk."""
        data = {
            "entries": [
                {
                    "question": e.question,
                    "teacher_context": e.teacher_context,
                    "rescued_model": e.rescued_model,
                    "source_model": e.source_model,
                    "original_trace": e.original_trace,
                    "rescue_trace": e.rescue_trace,
                    "timestamp": e.timestamp,
                }
                for e in self.buffer
            ],
            "stats": self.stats,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load buffer from disk."""
        with open(path) as f:
            data = json.load(f)

        self.buffer.clear()
        for entry_dict in data["entries"]:
            entry = BuddyEntry(**entry_dict)
            self.buffer.append(entry)

        self.stats = data.get("stats", self.stats)
