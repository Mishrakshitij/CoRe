"""Small deterministic tests of paper equations and protocol failure cases."""

from dataclasses import replace
import unittest

import numpy as np

from core.parsing import answers_equal, extract_answer, extract_strategies, normalize_answer
from core.protocol import build_hint, build_prompt, select_teacher, strip_explicit_answers
from core.rewards import (RewardConfig, Sample, dpp_lite_rewards, group_normalize,
    hybrid_distances, operation_signatures, overlap_partial_credit,
    score_question, score_question_groups)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)


def sample(model="m1", round_id="A", correct=False, *, text="Reasoning", **kwargs):
    return Sample("q1", model, round_id, text, "2" if correct else "3", correct, gold="2", **kwargs)


class ParsingTests(unittest.TestCase):
    def test_explicit_formats_and_latest_correction(self):
        for text in ["Steps\n#### 42", "Steps\nFinal answer: 42", "<final_answer>42</final_answer>",
                     "<final answer>42</final answer>", "**Final answer: 42**", r"\boxed{42}"]:
            with self.subTest(text=text):
                self.assertTrue(answers_equal(extract_answer(text), "42"))
        self.assertTrue(answers_equal(extract_answer("\\boxed{4}\nFinal answer: 5"), "5"))

    def test_no_unmarked_last_number_and_no_missing_success(self):
        self.assertIsNone(extract_answer("Try 5, then subtract 3."))
        self.assertIsNone(extract_answer("I cannot solve this."))
        self.assertIsNone(extract_answer("<answer>2</result>"))
        self.assertFalse(answers_equal(None, None))
        self.assertFalse(answers_equal("", ""))

    def test_numeric_normalization_conservative(self):
        for value in ["0.5", "1/2", r"\frac{1}{2}", r"\boxed{\frac{1}{2}}"]:
            self.assertEqual(normalize_answer(value), "1/2")
        self.assertTrue(answers_equal("1,000,000", "1000000.0"))
        self.assertTrue(answers_equal("[A]", "A"))
        self.assertFalse(answers_equal("50%", "0.5"))
        self.assertFalse(answers_equal("2 meters", "2"))
        self.assertFalse(answers_equal("-2", "2"))
        self.assertIsNone(extract_answer("1/0"))

    def test_strategy_outcome_tags(self):
        text = '<strategy id="1">Wrong path.<strategy_id_outcome>7</strategy_id_outcome></strategy>'
        text += '<strategy id="2">Add the terms.<strategy_id_outcome>2</strategy_id_outcome></strategy>'
        self.assertEqual([s.answer for s in extract_strategies(text)], ["7", "2"])
        self.assertEqual(build_hint(text, "2"), "Add the terms.")


class HintTests(unittest.TestCase):
    def test_teacher_correct_cold_and_configurable_score(self):
        wrong = sample(correct=False, reward=100)
        contexted = sample(round_id="B", correct=True, reward=20)
        first = sample(correct=True, reward=2, exploit_reward=1)
        second = sample(model="m2", correct=True, reward=1, exploit_reward=1.3)
        candidates = [wrong, contexted, first, second]
        self.assertIs(select_teacher(candidates), first)
        self.assertIs(select_teacher(candidates, score="exploit_reward"), second)
        self.assertIsNone(select_teacher([wrong, contexted]))

    def test_answer_payload_stripping_and_exact_cap(self):
        text = "Compute a ratio.\n<answer>123456</answer>\nFinal answer: 123456\n#### 123456\n\\boxed{\\frac{123456}{1}}"
        hint = build_hint(text, "123456")
        self.assertEqual(hint, "Compute a ratio.")
        self.assertNotIn("123456", hint)
        self.assertEqual(build_hint(text, "123456", 7, CharacterTokenizer()), "Compute")
        self.assertEqual(build_hint(text, "123456", 0), "")

    def test_missing_or_malformed_strategy_outcome_disables_hint(self):
        self.assertEqual(build_hint('<strategy id="1">Guess.</strategy>', "2"), "")
        self.assertEqual(build_hint("<answer>2</answer>", "2"), "")
        self.assertEqual(strip_explicit_answers("Reason. <answer>2"), "Reason.")
        for text, gold in [("42", "42"), ("[42]", "42"), ("1/2", "0.5"), ("A", "A"), ("[A]", "A")]:
            with self.subTest(text=text):
                self.assertEqual(build_hint(text, gold), "")

    def test_prompt_has_no_hint_when_empty(self):
        self.assertEqual(build_prompt("Find x."), "Question: Find x.\nLet's solve this step by step:")
        self.assertNotIn("Hint:", build_prompt("Find x.", ""))
        self.assertIn("Hint:\nSubstitute first.", build_prompt("Find x.", "Substitute first."))


class DistanceTests(unittest.TestCase):
    def test_hybrid_cosine_and_jaccard_equation(self):
        matrix = hybrid_distances(["add", "divide"], embeddings=np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(matrix, [[0, 1], [1, 0]])
        antipodal = hybrid_distances(["plain", "plain"], embeddings=np.array([[1, 0], [-1, 0]]))
        self.assertAlmostEqual(antipodal[0, 1], 1.2)

    def test_zero_nan_embeddings_rejected_and_empty_ops_equal(self):
        for vectors in [np.array([[0, 0]]), np.array([[np.nan, 1]])]:
            with self.assertRaises(ValueError):
                hybrid_distances(["plain"], embeddings=vectors)
        np.testing.assert_array_equal(hybrid_distances(["plain", "ordinary"],
            embedding_weight=0, structural_weight=1), np.zeros((2, 2)))
        self.assertNotIn("case_analysis", operation_signatures("different"))
        self.assertNotIn("comparison", operation_signatures('<strategy id="1">plain</strategy>'))


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.base = RewardConfig(w_explore=0, cross_model_weight=0)

    def test_literal_dpp_cap_and_degeneracy(self):
        distance = np.ones((3, 3)) - np.eye(3)
        rewards, selected, reason = dpp_lite_rewards(distance, [1, 0, 0], set_cap=1)
        np.testing.assert_allclose(rewards, [0, .85, .85])
        self.assertEqual(selected, (0,))
        self.assertEqual(reason, "cap")
        rewards, selected, reason = dpp_lite_rewards(distance, [1, 0, 0], set_cap=10)
        np.testing.assert_array_equal(rewards, [0, 0, 0])
        self.assertEqual(selected, (0, 1, 2))
        self.assertEqual(reason, "exhausted")
        duplicate = np.zeros((3, 3))
        rewards, selected, reason = dpp_lite_rewards(duplicate, [0, 1, 0])
        self.assertEqual(selected, (1,))
        self.assertEqual(reason, "margin")
        self.assertTrue(np.all(rewards == 0))

    def test_exploitation_callback_is_explicit_and_bounded(self):
        with self.assertRaises(ValueError):
            score_question([sample(correct=True)], replace(self.base, alpha=.3))
        result = [sample(correct=True), sample(correct=False)]
        score_question(result, replace(self.base, alpha=.3), partial_credit_fn=overlap_partial_credit)
        self.assertAlmostEqual(result[0].reward, 1.3)
        self.assertEqual(result[1].reward, 0)
        for invalid in [float("nan"), -1, 1.1]:
            with self.assertRaises(ValueError):
                score_question(result, replace(self.base, alpha=.3), partial_credit_fn=lambda *args: invalid)

    def test_rescue_requires_cold_failure_correct_hint_and_peer(self):
        samples = [sample("m1", correct=False), sample("m2", correct=True),
            sample("m1", "B", True, hint_provided=True),
            sample("m1", "B", True, hint_provided=False),
            sample("m2", "B", True, hint_provided=True),
            sample("m1", "B", False, hint_provided=True)]
        diagnostics = score_question(samples, self.base)
        self.assertEqual(diagnostics.components["rescue"], [0, 0, 1, 0, 0, 0])
        self.assertAlmostEqual(samples[2].reward, 1.15)
        no_cold = sample("unseen", "B", True, hint_provided=True)
        self.assertEqual(score_question([no_cold], self.base).components["rescue"], [0])
        failed = [sample(), sample("m2"), sample("m1", "B", True, hint_provided=True)]
        self.assertEqual(score_question(failed, self.base).components["rescue"], [0, 0, 0])

    def test_cross_gate_applies_to_peer_and_retains_all_peer_traces(self):
        config = replace(self.base, cross_model_weight=1, embedding_weight=1, structural_weight=0)
        items = [sample("m1", correct=False), sample("m2", correct=True)]
        result = score_question(items, config, embeddings=np.eye(2))
        np.testing.assert_allclose(result.components["cross_model"], [.85, 0])
        # A low-quality peer identical to the focal trace suppresses min-distance
        # even though its partner model passes the gate through another trace.
        items.append(sample("m2", correct=False))
        result = score_question(items, config, embeddings=np.array([[1, 0], [0, 1], [1, 0]]))
        self.assertEqual(result.components["cross_model"][0], 0)

    def test_trace_accuracy_uses_round_pool_and_normalization_uses_all(self):
        items = [sample("m1", correct=True), sample("m2", correct=False),
                 sample("m1", "B", True), sample("m2", "B", True)]
        result = score_question(items, replace(self.base, trace_accuracy_weight=1))
        np.testing.assert_allclose(result.components["trace_accuracy"], [.5, 0, 0, 0])
        expected = group_normalize([1.5, 0, 1, 1])
        np.testing.assert_allclose([s.advantage for s in items], expected)
        self.assertNotEqual(items[2].advantage, 0)  # independent B normalization would be zero

    def test_question_groups_do_not_leak_statistics(self):
        items = [sample(correct=True), sample(correct=False), replace(sample(correct=True), question_id="q2")]
        score_question_groups(items, self.base)
        self.assertAlmostEqual(items[0].advantage, 1, places=6)
        self.assertAlmostEqual(items[1].advantage, -1, places=6)
        self.assertEqual(items[2].advantage, 0)
        np.testing.assert_array_equal(group_normalize([2, 2]), [0, 0])
        with self.assertRaises(ValueError):
            score_question(items, self.base)

    def test_diagnostics_reports_default_zero_exploration(self):
        config = replace(self.base, w_explore=1, embedding_weight=0, structural_weight=1)
        result = score_question([sample(text="add"), sample("m2", text="divide")], config)
        self.assertTrue(result.exploration_all_zero)
        self.assertEqual(result.exploration_stop_reason, "exhausted")


if __name__ == "__main__":
    unittest.main()
