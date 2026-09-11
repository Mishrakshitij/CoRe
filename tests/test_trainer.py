"""Protocol regression tests run without model downloads or GPU dependencies."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from core.rewards import RewardConfig
from core.trainer import CoReTrainer, Example, Generation, ModelConfig, TrainConfig, grpo_loss, load_jsonl


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)


class FakePolicy:
    def __init__(self, model_id, events):
        self.model_id, self.events = model_id, events
        self.tokenizer = FakeTokenizer()
        self.count = 0
        self.observed = []

    def generate(self, prompt):
        self.events.append(("generate", self.model_id))
        self.count += 1
        correct = self.model_id == "M1" or self.count > 2
        answer = "4" if correct else "5"
        text = f'<strategy id="1">Add two and two.\n<answer>{answer}</answer></strategy>\n<final_answer>{answer}</final_answer>'
        return Generation(text, tuple(self.tokenizer.encode(prompt)), (10, 20, 30))

    def snapshot(self, rollout):
        self.events.append(("snapshot", self.model_id))
        rollout.old_log_probs = (-0.1, -0.2, -0.3)
        rollout.reference_log_probs = (-0.1, -0.2, -0.3)

    def update(self, rollouts):
        self.events.append(("update", self.model_id))
        for rollout in rollouts:
            assert rollout.old_log_probs is not None
            assert rollout.reference_log_probs is not None
            assert tuple(self.tokenizer.encode(rollout.prompt)) == rollout.generation.prompt_ids
        self.observed.extend(rollouts)
        return {"loss": 0.0}

    def save(self, path):
        path.mkdir(parents=True)
        (path / "fake-adapter.json").write_text("{}")


def config(**kwargs):
    defaults = dict(models=[ModelConfig("M1", "fake"), ModelConfig("M2", "fake")],
                    epochs=1, hint_probability=1.0)
    defaults.update(kwargs)
    return TrainConfig(**defaults)


class TrainerProtocolTests(unittest.TestCase):
    def test_two_rounds_keep_prompts_and_snapshot_team_before_update(self):
        events = []
        backends = [FakePolicy("M1", events), FakePolicy("M2", events)]
        trainer = CoReTrainer(config(), backends, encoder=None)
        rewards = RewardConfig(w_explore=0, cross_model_weight=0)
        rollouts, metrics = trainer.train_batch([Example("q1", "What is 2 + 2?", "4")], rewards)
        self.assertEqual(len(rollouts), 6)
        first_update = next(i for i, event in enumerate(events) if event[0] == "update")
        self.assertEqual(sum(event[0] == "snapshot" for event in events[:first_update]), 6)
        self.assertFalse(any(event[0] != "update" for event in events[first_update:]))
        self.assertEqual(metrics["rescue_eligible"], 1)
        self.assertEqual(metrics["rescued"], 1)
        m2_cold = [r for r in rollouts if r.sample.model_id == "M2" and r.sample.round_id == "A"]
        m2_context = next(r for r in rollouts if r.sample.model_id == "M2" and r.sample.round_id == "B")
        self.assertNotEqual(m2_context.prompt, m2_cold[0].prompt)
        self.assertTrue(m2_context.sample.hint_provided)
        self.assertAlmostEqual(sum(r.sample.advantage for r in rollouts), 0.0)
        # A model with two wrong A traces must still receive negative advantages
        # relative to its successful peer; per-model normalization would erase them.
        self.assertTrue(all(r.sample.advantage < 0 for r in m2_cold))

    def test_hint_dropout_prevents_rescue_credit(self):
        events = []
        trainer = CoReTrainer(config(hint_probability=0), [FakePolicy("M1", events), FakePolicy("M2", events)], None)
        rollouts, metrics = trainer.train_batch([Example("q1", "What is 2 + 2?", "4")],
                                               RewardConfig(w_explore=0, cross_model_weight=0))
        self.assertEqual(metrics["rescue_eligible"], 0)
        self.assertFalse(any(r.sample.hint_provided for r in rollouts))

    def test_question_advantages_are_not_mixed_across_batch(self):
        events = []
        trainer = CoReTrainer(config(), [FakePolicy("M1", events), FakePolicy("M2", events)], None)
        rollouts, _ = trainer.train_batch([Example("q1", "What is 2 + 2?", "4"),
                                          Example("q2", "What is 2 + 3?", "5")],
                                         RewardConfig(w_explore=0, cross_model_weight=0))
        for question_id in ("q1", "q2"):
            self.assertAlmostEqual(sum(r.sample.advantage for r in rollouts if r.sample.question_id == question_id), 0)

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GRPO only"):
            config(algorithm="gspo").validate()

    def test_output_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "keep.txt").write_text("prior run")
            trainer = CoReTrainer(config(output_dir=temporary), [FakePolicy("M1", []), FakePolicy("M2", [])], None)
            with self.assertRaises(FileExistsError):
                trainer.fit([Example("q", "Q", "4")])
            self.assertEqual((Path(temporary) / "keep.txt").read_text(), "prior run")

    def test_duplicate_data_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            row = json.dumps({"id": "q", "question": "Q", "answer": "4"}) + "\n"
            path.write_text(row + row)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_jsonl(path)

    def test_invalid_ids_and_nonfinite_answers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            for identity, answer in ((None, "4"), (" ", "4"), (True, "4"), ("q", float("nan")), ("q", float("inf"))):
                with self.subTest(identity=identity, answer=answer):
                    path.write_text(json.dumps({"id": identity, "question": "Q", "answer": answer}) + "\n")
                    with self.assertRaises(ValueError):
                        load_jsonl(path)

    def test_partial_credit_configuration_checked_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, "alpha=0"):
            config(partial_credit="none", reward={"alpha": 0.3}).validate()


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is optional for offline protocol tests")
class GRPOLossTests(unittest.TestCase):
    def test_negative_advantage_uses_minimum_and_snapshots_detach(self):
        import torch

        current = torch.tensor([-3.0], requires_grad=True)
        old = torch.tensor([-2.0], requires_grad=True)
        reference = torch.tensor([-2.0], requires_grad=True)
        loss, _ = grpo_loss(current, old, reference, -1.0, 0.2, 0.2, 0.0)
        self.assertAlmostEqual(loss.item(), 0.8, places=6)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)
        self.assertAlmostEqual(current.grad.item(), 0.0)

    def test_on_policy_positive_advantage_has_learning_gradient(self):
        import torch

        current = torch.tensor([-2.0, -3.0], requires_grad=True)
        loss, kl = grpo_loss(current, current.detach(), current.detach(), 1.0, 0.2, 0.2, 0.04)
        loss.backward()
        self.assertEqual(kl.item(), 0.0)
        self.assertTrue(torch.all(current.grad < 0).item())


if __name__ == "__main__":
    unittest.main()
