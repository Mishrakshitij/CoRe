"""Entry point: python -m core.cli --config configs/paper_qwen.json --data train.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trainer import TrainConfig, load_jsonl, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CoRe paper reference implementation")
    parser.add_argument("--config", type=Path, required=True, help="JSON training configuration")
    parser.add_argument("--data", type=Path, required=True, help="JSONL with id, question, answer; provide your training split")
    parser.add_argument("--output-dir", type=str, help="Override configuration output_dir")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration and data without loading models")
    args = parser.parse_args()
    try:
        config = TrainConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        if args.output_dir:
            config.output_dir = args.output_dir
        examples = load_jsonl(args.data)
        if args.validate_only:
            print(json.dumps({"valid": True, "examples": len(examples),
                              "models": [model.model_id for model in config.models],
                              "algorithm": config.algorithm, "output_dir": config.output_dir}))
            return
        run_training(config, examples)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    except ImportError as error:
        parser.error(f"Missing training dependency: {error}. Install requirements-paper.txt.")


if __name__ == "__main__":
    main()
