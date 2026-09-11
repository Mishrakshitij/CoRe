"""Cold, example-aligned individual and oracle team evaluation.

One JSONL row describes one model on one example. Pass@k uses the first k
recorded samples, not the combinatorial estimator from a larger sample pool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Iterable


def evaluate_records(records: Iterable[dict], k: int = 2) -> dict:
    """Evaluate aligned cold samples; reject missing or ambiguous identities."""
    from .parsing import extract_answer, normalize_answer, answers_equal

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    by_model = defaultdict(dict)
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Each prediction row must be an object")
        for field in ("example_id", "model", "gold", "completions"):
            if field not in row:
                raise ValueError(f"Missing prediction field: {field}")
        example_id, model = row["example_id"], row["model"]
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("example_id must be a nonempty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if example_id in by_model[model]:
            raise ValueError(f"Duplicate prediction for {model}/{example_id}")
        if row.get("round", "A") != "A" or row.get("hint_used", False):
            raise ValueError("Cold evaluation requires Round A samples without hints")
        gold = row["gold"]
        if (not isinstance(gold, (str, int, float)) or isinstance(gold, bool)
                or (isinstance(gold, float) and not math.isfinite(gold))
                or not normalize_answer(str(gold))):
            raise ValueError("gold must be a nonempty answer")
        samples = row["completions"]
        if not isinstance(samples, list) or len(samples) < k or not all(isinstance(s, str) for s in samples):
            raise ValueError(f"{model}/{example_id} requires at least {k} string completions")
        counts = row.get("token_counts")
        if counts is not None and (
            not isinstance(counts, list) or len(counts) != len(samples)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts)
        ):
            raise ValueError("token_counts must contain one nonnegative integer per completion")
        by_model[model][example_id] = row
    if not by_model:
        raise ValueError("No predictions supplied")
    models = sorted(by_model)
    ids = sorted(by_model[models[0]])
    for model in models:
        if set(by_model[model]) != set(ids):
            raise ValueError("All models must contain exactly the same example IDs")
    per_model_hits = {model: [] for model in models}
    team_hits, votes, per_example = [], [], []
    all_counts = []
    complete_counts = True
    for example_id in ids:
        gold = str(by_model[models[0]][example_id]["gold"])
        answers = []
        solved = {}
        for model in models:
            row = by_model[model][example_id]
            if not answers_equal(str(row["gold"]), gold):
                raise ValueError(f"Conflicting gold answers for example {example_id}")
            parsed = [extract_answer(s) for s in row["completions"][:k]]
            solved[model] = any(answers_equal(a, gold) for a in parsed)
            per_model_hits[model].append(solved[model])
            answers.extend(normalize_answer(a) for a in parsed if a)
            if row.get("token_counts") is None:
                complete_counts = False
            else:
                all_counts.extend(row["token_counts"][:k])
        # Counter preserves arrival order: ties resolve by sorted model, sample index.
        majority = Counter(answers).most_common(1)[0][0] if answers else ""
        majority_correct = answers_equal(majority, gold)
        team_correct = any(solved.values())
        votes.append(majority_correct)
        team_hits.append(team_correct)
        per_example.append({"example_id": example_id, "model_solved": solved,
                            "oracle_team_solved": team_correct,
                            "majority_answer": majority, "majority_correct": majority_correct})
    scores = {model: sum(hits) / len(ids) for model, hits in per_model_hits.items()}
    team_score = sum(team_hits) / len(ids)
    return {
        "num_examples": len(ids), "num_models": len(models), "k_per_model": k,
        "team_samples_per_example": k * len(models),
        "individual_pass_at_k": scores, "oracle_team_pass_at_k": team_score,
        "collaboration_gain_over_best_individual": team_score - max(scores.values()),
        "majority_vote_accuracy": sum(votes) / len(ids),
        "majority_tie_break": "first in sorted model name, then sample order",
        "mean_completion_tokens": sum(all_counts) / len(all_counts) if complete_counts and all_counts else None,
        "max_observed_completion_tokens": max(all_counts) if complete_counts and all_counts else None,
        "generation_budget_validation": "not verified from precomputed completions; retain generation config",
        "evaluation_mode": "cold; first k recorded samples; oracle team selection uses gold",
        "per_example": per_example,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSONL with example_id/model/gold/completions")
    parser.add_argument("--k", type=int, default=2, help="samples per model, default: 2")
    parser.add_argument("--output", type=Path, help="optional metrics JSON output")
    args = parser.parse_args(argv)
    try:
        with args.predictions.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        result = evaluate_records(records, args.k)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
