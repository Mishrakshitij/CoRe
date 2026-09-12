"""Generate cold evaluation samples from base models or exported LoRA adapters."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="held-out JSONL: id, question, answer")
    parser.add_argument("--output", type=Path, required=True, help="new predictions JSONL; existing files are preserved")
    parser.add_argument("--checkpoint-dir", type=Path, help="export directory containing one adapter folder per model_id")
    parser.add_argument("--k", type=int, default=2, help="samples per model")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true", help="greedy Pass@1; requires --k 1")
    parser.add_argument("--validate-only", action="store_true", help="validate configuration/data without loading models")
    args = parser.parse_args(argv)
    from .trainer import TrainConfig, load_jsonl, load_model_and_tokenizer, encode_prompt
    from .protocol import build_prompt
    try:
        config = TrainConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        examples = load_jsonl(args.data)
        if args.k < 1 or args.max_new_tokens < 1 or args.seed < 0:
            raise ValueError("k/max-new-tokens must be positive and seed nonnegative")
        if args.greedy and args.k != 1:
            raise ValueError("Greedy evaluation requires --k 1")
        if args.checkpoint_dir:
            for spec in config.models:
                if not (args.checkpoint_dir / spec.model_id / "adapter_config.json").is_file():
                    raise ValueError(f"Missing adapter: {args.checkpoint_dir / spec.model_id}")
        manifest_path = args.output.with_name(args.output.name + ".meta.json")
        if args.output.exists() or manifest_path.exists():
            raise ValueError("Output or metadata already exists; choose a new output filename")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({"examples": len(examples), "models": len(config.models),
                          "k_per_model": args.k, "max_new_tokens": args.max_new_tokens,
                          "mode": "cold greedy" if args.greedy else "cold stochastic"}, indent=2))
        return 0
    try:
        import torch
        from transformers import set_seed
        from peft import PeftModel
    except ImportError as exc:
        parser.error(f"Install requirements-paper.txt to generate samples: {exc}")
    packages = {}
    for package in ("torch", "transformers", "peft", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    metadata = {"status": "incomplete", "config": asdict(config), "seed": args.seed,
                "k_per_model": args.k, "max_new_tokens": args.max_new_tokens,
                "decoding": "greedy" if args.greedy else "stochastic",
                "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
                "example_ids": [example.id for example in examples], "packages": packages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    dtype = getattr(torch, config.dtype)
    with args.output.open("x", encoding="utf-8") as destination:
        for model_index, spec in enumerate(config.models):
            set_seed(args.seed + model_index)
            tokenizer_source = str(args.checkpoint_dir / spec.model_id) if args.checkpoint_dir else None
            model, tokenizer = load_model_and_tokenizer(spec, dtype, tokenizer_source)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                raise ValueError(f"{spec.model_id} tokenizer has neither pad nor EOS token")
            if args.checkpoint_dir:
                model = PeftModel.from_pretrained(model, str(args.checkpoint_dir / spec.model_id))
            model.eval()
            for example in examples:
                prompt = build_prompt(example.question)
                ids = encode_prompt(tokenizer, prompt)
                if not ids:
                    raise ValueError(f"Tokenizer produced an empty prompt for {example.id}")
                input_ids = torch.tensor([ids], dtype=torch.long, device=spec.device)
                inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
                prompt_length = len(ids)
                if prompt_length > config.max_prompt_tokens:
                    raise ValueError(f"Prompt {example.id} exceeds max_prompt_tokens; no silent truncation")
                text_config = getattr(model.config, "text_config", model.config)
                context_limit = getattr(text_config, "max_position_embeddings", None)
                if context_limit and prompt_length + args.max_new_tokens > context_limit:
                    raise ValueError(f"Prompt plus completion budget exceeds model context for {example.id}")
                completions, counts = [], []
                for _ in range(args.k):
                    sampling = {"do_sample": not args.greedy}
                    if not args.greedy:
                        sampling.update(temperature=config.temperature, top_p=config.top_p, top_k=0)
                    with torch.inference_mode():
                        sequence = model.generate(
                            **inputs, max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id, **sampling,
                        )[0, prompt_length:]
                    completions.append(tokenizer.decode(sequence, skip_special_tokens=True))
                    counts.append(int(sequence.numel()))
                record = {"example_id": example.id, "model": spec.model_id, "gold": example.answer,
                          "completions": completions, "token_counts": counts, "round": "A", "hint_used": False}
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metadata["status"] = "complete"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved cold predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
