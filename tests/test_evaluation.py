"""Regression tests for team budget accounting and example alignment."""

import unittest
from core.evaluation import evaluate_records


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"example_id": "x", "model": "a", "gold": "2", "completions": ["<final_answer>2</final_answer>", "<final_answer>0</final_answer>"], "token_counts": [5, 6]},
            {"example_id": "y", "model": "a", "gold": "3", "completions": ["<final_answer>0</final_answer>", "<final_answer>0</final_answer>"], "token_counts": [5, 6]},
            {"example_id": "y", "model": "b", "gold": "3", "completions": ["<final_answer>3</final_answer>", "<final_answer>3</final_answer>"], "token_counts": [5, 6]},
            {"example_id": "x", "model": "b", "gold": "2", "completions": ["<final_answer>0</final_answer>", "<final_answer>0</final_answer>"], "token_counts": [5, 6]},
        ]

    def test_alignment_and_oracle_budget(self):
        result = evaluate_records(self.rows, 2)
        self.assertEqual(result["individual_pass_at_k"], {"a": .5, "b": .5})
        self.assertEqual(result["oracle_team_pass_at_k"], 1.)
        self.assertEqual(result["team_samples_per_example"], 4)
        self.assertEqual(result["collaboration_gain_over_best_individual"], .5)
        self.assertEqual(result["mean_completion_tokens"], 5.5)
        self.assertLess(result["majority_vote_accuracy"], result["oracle_team_pass_at_k"])

    def test_reject_incomplete_or_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "same example IDs"):
            evaluate_records(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_records(self.rows + [self.rows[0]])

    def test_reject_different_gold_or_insufficient_samples(self):
        rows = [dict(row) for row in self.rows]
        rows[-1]["gold"] = "8"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            evaluate_records(rows)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            evaluate_records(self.rows, 3)

    def test_reject_hint_evaluation(self):
        rows = [dict(row) for row in self.rows]
        rows[-1]["hint_used"] = True
        with self.assertRaisesRegex(ValueError, "Cold evaluation"):
            evaluate_records(rows)

    def test_missing_counts_never_become_estimates(self):
        rows = [dict(row) for row in self.rows]
        rows[-1].pop("token_counts")
        self.assertIsNone(evaluate_records(rows)["mean_completion_tokens"])

    def test_nonfinite_gold_rejected(self):
        for value in (float("nan"), float("inf")):
            rows = [dict(row) for row in self.rows]
            rows[-1]["gold"] = value
            with self.assertRaisesRegex(ValueError, "gold"):
                evaluate_records(rows)


if __name__ == "__main__":
    unittest.main()
