"""
Collaborative Trainer
=====================

Main trainer for collaborative reasoning with GRPO/GSPO/SAPO.

Implements:
- Pairwise model collaboration
- Multi-model collaboration (N > 2)
- Two-epoch training with annealing
- Micro-round A+B training
- Buddy buffer distillation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .base_trainer import BaseCollabTrainer, TrainingState
from .micro_rounds import MicroRoundManager, MicroRoundOutput
from .buddy_buffer import BuddyBuffer
from ..losses import get_loss_fn, LossOutput
from ..rewards import CombinedRewardFunction


logger = logging.getLogger(__name__)


class CollaborativeTrainer(BaseCollabTrainer):
    """
    Full collaborative trainer with micro-round support.

    Training flow per question:
    1. Micro-round A: All models generate K traces independently
    2. Identify best correct trace across models
    3. Compress to teacher context
    4. Micro-round B: All models generate K' traces with context (hint dropout)
    5. Compute rewards (with rescue bonus)
    6. GRPO/GSPO/SAPO update for each model
    7. Store successful rescues in buddy buffer
    """

    def __init__(
        self,
        config: dict,
        model_configs: List[dict],
        output_dir: str,
    ):
        super().__init__(config, model_configs, output_dir)

        # Micro-round manager
        self.micro_round = MicroRoundManager(config)

        # Buddy buffer
        self.buddy_buffer = BuddyBuffer(
            max_size=config["collaboration"]["max_buffer_size"]
        )

        # Distillation settings
        self.distillation_batch_size = config["collaboration"]["distillation_batch_size"]
        self.distillation_frequency = config["collaboration"]["distillation_frequency"]

        # Metrics tracking
        self.epoch_metrics = {
            "accuracy": [],
            "rescue_rate": [],
            "diversity": [],
            "loss": [],
        }

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
        """
        # Load models
        self.load_all_models()
        self.setup_optimizers()

        # Calculate total steps
        num_training_steps = (
            len(train_dataloader) * self.num_epochs
            // self.gradient_accumulation_steps
        )
        self.setup_schedulers(num_training_steps)

        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Models: {list(self.models.keys())}")

        for epoch in range(1, self.num_epochs + 1):
            self.state.epoch = epoch
            self.reward_fn.set_epoch(epoch)

            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")

            # Training epoch
            self.train_epoch(train_dataloader, epoch)

            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(f"Eval metrics: {eval_metrics}")

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}")

            # Epoch 2: Add distillation from buddy buffer
            if epoch == 2 and len(self.buddy_buffer) > 0:
                logger.info("Running distillation from buddy buffer...")
                self.distillation_epoch(train_dataloader)

        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training complete!")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ):
        """Train for one epoch."""
        # Set models to train mode
        for model in self.models.values():
            model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_rescue_rate = 0.0
        num_rescues = 0
        num_rescue_attempts = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            questions = batch["question"]
            ground_truths = batch["answer"]

            # Process batch (one question at a time for collaborative training)
            for q_idx, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
                # Train step for this question
                step_metrics = self.train_step(
                    question=question,
                    ground_truth=ground_truth,
                    step=self.state.global_step,
                )

                # Accumulate metrics
                epoch_loss += step_metrics.get("loss", 0)
                epoch_accuracy += step_metrics.get("accuracy", 0)

                if step_metrics.get("rescue_attempt", False):
                    num_rescue_attempts += 1
                    if step_metrics.get("rescue_success", False):
                        num_rescues += 1

                self.state.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{step_metrics.get('loss', 0):.4f}",
                    "acc": f"{step_metrics.get('accuracy', 0):.2%}",
                    "rescue": f"{num_rescues}/{num_rescue_attempts}",
                })

                # Periodic logging
                if self.state.global_step % self.config["training"]["logging_steps"] == 0:
                    self.log_metrics(step_metrics, self.state.global_step)

                # Periodic evaluation
                if self.state.global_step % self.config["training"]["eval_steps"] == 0:
                    # Could add periodic eval here
                    pass

                # Periodic checkpoint
                if self.state.global_step % self.config["training"]["save_steps"] == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

        # Epoch summary
        num_samples = len(dataloader) * self.batch_size
        epoch_rescue_rate = num_rescues / max(1, num_rescue_attempts)

        logger.info(f"Epoch {epoch} summary:")
        logger.info(f"  Average loss: {epoch_loss / num_samples:.4f}")
        logger.info(f"  Average accuracy: {epoch_accuracy / num_samples:.2%}")
        logger.info(f"  Rescue rate: {epoch_rescue_rate:.2%} ({num_rescues}/{num_rescue_attempts})")
        logger.info(f"  Buddy buffer size: {len(self.buddy_buffer)}")

    def train_step(
        self,
        question: str,
        ground_truth: str,
        step: int,
    ) -> Dict[str, float]:
        """
        Train step for a single question.

        Implements full micro-round A+B training flow.
        """
        metrics = {}

        # ========== Micro-round A: Cold Generation ==========
        round_a_results = {}
        any_correct = False
        best_correct_trace = None
        best_correct_model = None

        for model_id in self.models:
            # Generate K traces without context
            traces = self.generate_traces(
                model_id=model_id,
                questions=[question],
                num_traces=self.K,
            )[0]  # Single question

            # Process round A
            round_a = self.micro_round.process_round_a(
                traces=traces,
                ground_truth=ground_truth,
                question=question,
                reward_fn=self.reward_fn,
            )
            round_a_results[model_id] = round_a

            # Track best correct trace
            if round_a.best_correct_trace is not None:
                any_correct = True
                if best_correct_trace is None:
                    best_correct_trace = round_a.best_correct_trace
                    best_correct_model = model_id

        metrics["round_a_any_correct"] = float(any_correct)

        # ========== Micro-round B: Contexted Generation ==========
        round_b_results = {}
        rescue_success = False

        if best_correct_trace is not None:
            # Compress to teacher context
            teacher_context = self.micro_round.build_teacher_context(best_correct_trace)

            for model_id in self.models:
                # Generate K' traces with hint dropout
                traces_b = []
                used_hint = []

                for _ in range(self.K_prime):
                    use_hint = self.micro_round.should_use_hint()
                    used_hint.append(use_hint)

                    if use_hint:
                        prompt = self.micro_round.format_contexted_prompt(
                            question, teacher_context
                        )
                    else:
                        prompt = question

                    trace = self.generate_traces(
                        model_id=model_id,
                        questions=[prompt if use_hint else question],
                        num_traces=1,
                        context=teacher_context if use_hint else None,
                    )[0][0]
                    traces_b.append(trace)

                # Process round B
                round_a_had_correct = round_a_results[model_id].best_correct_trace is not None
                round_b = self.micro_round.process_round_b(
                    traces=traces_b,
                    ground_truth=ground_truth,
                    question=question,
                    reward_fn=self.reward_fn,
                    used_hint=used_hint,
                    round_a_had_correct=round_a_had_correct,
                )
                round_b_results[model_id] = round_b

                # Check for rescue
                if round_b.rescue_success:
                    rescue_success = True
                    # Add to buddy buffer
                    self.buddy_buffer.add(
                        question=question,
                        teacher_context=teacher_context,
                        rescued_model=model_id,
                        source_model=best_correct_model,
                        original_trace=round_a_results[model_id].traces[0] if round_a_results[model_id].traces else "",
                        rescue_trace=traces_b[0] if traces_b else "",
                        step=step,
                    )

        metrics["rescue_attempt"] = not any_correct or best_correct_trace is not None
        metrics["rescue_success"] = rescue_success

        # ========== GRPO/GSPO/SAPO Update ==========
        total_loss = 0.0
        total_accuracy = 0.0

        for model_id in self.models:
            # Combine round A and B traces
            round_a = round_a_results[model_id]
            round_b = round_b_results.get(model_id)

            all_traces, all_rewards, all_sources = self.micro_round.combine_rounds(
                round_a, round_b, lambda_b=0.8
            )

            if not all_traces:
                continue

            # Prepare prompts
            prompts = [f"Question: {question}\n\nLet's solve this step by step:"] * len(all_traces)

            # Compute log probs
            model = self.models[model_id]
            ref_model = self.ref_models[model_id]
            tokenizer = self.tokenizers[model_id]

            log_probs, mask = self.compute_log_probs(model, tokenizer, prompts, all_traces)
            with torch.no_grad():
                old_log_probs, _ = self.compute_log_probs(model, tokenizer, prompts, all_traces)
                ref_log_probs, _ = self.compute_log_probs(ref_model, tokenizer, prompts, all_traces)

            # Normalize advantages
            rewards_tensor = torch.tensor(all_rewards, device=log_probs.device)
            advantages = self.loss_fn.normalize_advantages(rewards_tensor)

            # Compute loss
            is_rescue = torch.tensor(
                [s == 'contexted' and not any_correct for s in all_sources],
                device=log_probs.device,
            )

            loss_output = self.loss_fn(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                advantages=advantages,
                mask=mask,
                is_rescue=is_rescue if hasattr(self.loss_fn, 'forward') else None,
            )

            # Backward
            loss = loss_output.loss / self.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()

            # Accuracy for this model
            model_accuracy = sum(round_a.is_correct) / len(round_a.is_correct) if round_a.is_correct else 0
            total_accuracy += model_accuracy

            # Gradient accumulation step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizers[model_id].step()
                self.schedulers[model_id].step()
                self.optimizers[model_id].zero_grad()

        # Average metrics across models
        num_models = len(self.models)
        metrics["loss"] = total_loss / num_models
        metrics["accuracy"] = total_accuracy / num_models

        return metrics

    def distillation_epoch(self, dataloader: DataLoader):
        """
        Run distillation training using buddy buffer.

        Uses successful rescue contexts to train models
        that struggled with certain questions.
        """
        logger.info(f"Distillation with {len(self.buddy_buffer)} buffer entries")

        # Sample from buddy buffer
        samples = self.buddy_buffer.sample_for_distillation(
            batch_size=self.distillation_batch_size * 10
        )

        for model_id, entries in samples.items():
            if not entries:
                continue

            model = self.models[model_id]
            tokenizer = self.tokenizers[model_id]

            logger.info(f"Distillation for {model_id}: {len(entries)} samples")

            for entry in tqdm(entries, desc=f"Distill {model_id}"):
                # Create training example: question + context -> good trace
                prompt = self.micro_round.format_contexted_prompt(
                    entry.question,
                    entry.teacher_context,
                )

                # Generate target trace (or use stored rescue trace)
                target_trace = entry.rescue_trace if entry.rescue_trace else None

                if target_trace is None:
                    # Generate a new trace with context
                    traces = self.generate_traces(
                        model_id=model_id,
                        questions=[entry.question],
                        num_traces=1,
                        context=entry.teacher_context,
                    )
                    target_trace = traces[0][0] if traces and traces[0] else None

                if target_trace is None:
                    continue

                # Supervised fine-tuning loss
                inputs = tokenizer(
                    prompt + target_trace,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(model.device)

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                # Step optimizer
                self.optimizers[model_id].step()
                self.optimizers[model_id].zero_grad()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate all models on the dataset."""
        # Set models to eval mode
        for model in self.models.values():
            model.eval()

        metrics = {model_id: {"correct": 0, "total": 0} for model_id in self.models}
        combined_correct = 0
        combined_total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                questions = batch["question"]
                ground_truths = batch["answer"]

                for question, ground_truth in zip(questions, ground_truths):
                    any_correct = False

                    for model_id in self.models:
                        # Generate single trace for evaluation
                        traces = self.generate_traces(
                            model_id=model_id,
                            questions=[question],
                            num_traces=1,
                        )[0]

                        # Check correctness
                        result = self.reward_fn.exploit_reward(
                            traces[0], ground_truth, question
                        )

                        metrics[model_id]["total"] += 1
                        if result.is_correct:
                            metrics[model_id]["correct"] += 1
                            any_correct = True

                    combined_total += 1
                    if any_correct:
                        combined_correct += 1

        # Compute final metrics
        eval_metrics = {}
        for model_id, m in metrics.items():
            acc = m["correct"] / max(1, m["total"])
            eval_metrics[f"{model_id}_accuracy"] = acc

        eval_metrics["combined_accuracy"] = combined_correct / max(1, combined_total)
        eval_metrics["collaboration_gain"] = (
            eval_metrics["combined_accuracy"] -
            max(eval_metrics[f"{mid}_accuracy"] for mid in self.models)
        )

        # Set models back to train mode
        for model in self.models.values():
            model.train()

        return eval_metrics
