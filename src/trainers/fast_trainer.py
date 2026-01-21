"""
Fast Collaborative Trainer
==========================

Optimized trainer using:
1. Batched generation (2-3x speedup)
2. torch.compile for forward passes (1.3-1.5x speedup)
3. Memory-efficient gradient checkpointing
4. Optimized data loading

Total expected speedup: 3-5x vs standard training
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .base_trainer import BaseCollabTrainer, TrainingState
from .micro_rounds import MicroRoundManager
from .buddy_buffer import BuddyBuffer
from ..losses import get_loss_fn
from ..rewards import CombinedRewardFunction
from ..data.preprocessing import (
    format_prompt,
    format_multi_strategy_prompt,
    format_mistral3_prompt,
    format_phi4_prompt,
    is_mistral3_model,
    is_phi4_model,
)


logger = logging.getLogger(__name__)


class FastCollaborativeTrainer(BaseCollabTrainer):
    """
    Optimized collaborative trainer with batched generation.

    Key optimizations:
    1. Generate traces for multiple questions at once
    2. Use torch.compile for faster forward passes
    3. Efficient memory management
    """

    def __init__(
        self,
        config: dict,
        model_configs: List[dict],
        output_dir: str,
        use_compile: bool = True,
    ):
        super().__init__(config, model_configs, output_dir)

        # Store model configs dict for lookup by model_id
        self.model_configs_dict = {}
        for i, mc in enumerate(model_configs):
            model_id = f"M{i+1}"
            self.model_configs_dict[model_id] = mc

        # Optimization flags
        self.use_compile = use_compile

        # Micro-round manager
        self.micro_round = MicroRoundManager(config)

        # Buddy buffer
        self.buddy_buffer = BuddyBuffer(
            max_size=config["collaboration"]["max_buffer_size"]
        )

        # Distillation settings (matching original trainer)
        self.distillation_batch_size = config["collaboration"]["distillation_batch_size"]
        self.distillation_frequency = config["collaboration"]["distillation_frequency"]

        # Generation batch size (larger = faster but more memory)
        self.gen_batch_size = config.get("fast_training", {}).get("gen_batch_size", 4)
        self.log_round_rewards = config.get("fast_training", {}).get("log_round_rewards", False)
        self.log_round_reward_components = config.get("fast_training", {}).get(
            "log_round_reward_components",
            False,
        )

        # Metrics tracking
        self.epoch_metrics = {
            "accuracy": [],
            "rescue_rate": [],
            "loss": [],
        }

    def load_model(
        self,
        model_name: str,
        model_id: str,
        use_lora: bool = True,
    ):
        """Load model with optional torch.compile."""
        model, tokenizer = super().load_model(model_name, model_id, use_lora)

        # Apply torch.compile for faster inference (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                logger.info(f"Applying torch.compile to {model_id}...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info(f"  torch.compile applied successfully")
            except Exception as e:
                logger.warning(f"  torch.compile failed: {e}, continuing without compilation")

        return model, tokenizer

    def _format_prompt_for_model(
        self,
        model_id: str,
        question: str,
        context: str = None,
    ) -> str:
        """
        Format prompt based on template_type and model.

        Supports:
        - standard: Raw prompts (default)
        - mistral-chat: Chat template for Mistral-3 reasoning models
        - phi-chat: Chat template for Phi-4 reasoning models
        - auto: Auto-detect based on model name
        - context_template: Optional override for contexted prompts (standard/chat/auto)

        Args:
            model_id: Model identifier
            question: The question to solve
            context: Optional teacher context for contexted generation

        Returns:
            Formatted prompt string
        """
        tokenizer = self.tokenizers[model_id]
        model_name = self.model_configs_dict.get(model_id, {}).get("name", "")
        template_type = self.config.get("prompting", {}).get("template_type", "standard")
        multi_strategy = self.config.get("prompting", {}).get("multi_strategy", False)
        dataset = self.config.get("training", {}).get("dataset", "gsm8k")
        context_template = self.config.get("prompting", {}).get("context_template", "standard")
        use_full_trace_hint = self.config.get("collaboration", {}).get("use_full_trace_hint", False)
        hint_prefix = self.config.get("collaboration", {}).get("hint_prefix", None)

        if context:
            # Contexted generation with teacher hint (optional chat template)
            effective_context_template = context_template
            if context_template == "auto":
                if is_phi4_model(model_name):
                    effective_context_template = "phi-chat"
                elif is_mistral3_model(model_name):
                    effective_context_template = "mistral-chat"
                else:
                    effective_context_template = "standard"
            elif context_template == "chat":
                if is_phi4_model(model_name):
                    effective_context_template = "phi-chat"
                elif is_mistral3_model(model_name):
                    effective_context_template = "mistral-chat"
                else:
                    effective_context_template = "standard"

            if effective_context_template in ("phi-chat", "mistral-chat"):
                if hint_prefix is None:
                    if use_full_trace_hint:
                        hint_prefix = (
                            "A peer model solved this correctly. "
                            "Use its reasoning as guidance and follow a similar approach."
                        )
                    else:
                        hint_prefix = "A peer model provided this helpful approach:"
                contexted_question = (
                    f"{hint_prefix}\n\n"
                    f"<peer_hint>\n{context}\n</peer_hint>\n\n"
                    f"{question}"
                )
            else:
                if hint_prefix is None and use_full_trace_hint:
                    hint_prefix = "Use the following reasoning trace as guidance."
                if hint_prefix:
                    return f"{hint_prefix}\n\n{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"
                return f"{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"

            if effective_context_template == "phi-chat" and is_phi4_model(model_name):
                return format_phi4_prompt(
                    question=contexted_question,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    multi_strategy=multi_strategy,
                )

            if effective_context_template == "mistral-chat" and is_mistral3_model(model_name):
                return format_mistral3_prompt(
                    question=contexted_question,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    multi_strategy=multi_strategy,
                )

            # Default: plain context + question (current behavior)
            return f"{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"

        # Auto-detect template type based on model name
        if template_type == "auto":
            if is_phi4_model(model_name):
                template_type = "phi-chat"
            elif is_mistral3_model(model_name):
                template_type = "mistral-chat"
            else:
                template_type = "standard"

        # Check if we should use Phi-4 chat template
        if template_type == "phi-chat" and is_phi4_model(model_name):
            return format_phi4_prompt(
                question=question,
                tokenizer=tokenizer,
                dataset=dataset,
                multi_strategy=multi_strategy,
            )

        # Check if we should use Mistral chat template
        if template_type == "mistral-chat" and is_mistral3_model(model_name):
            return format_mistral3_prompt(
                question=question,
                tokenizer=tokenizer,
                dataset=dataset,
                multi_strategy=multi_strategy,
            )

        # Standard prompt formatting
        if multi_strategy:
            return format_multi_strategy_prompt(question, dataset=dataset)
        else:
            return format_prompt(question)

    @torch.no_grad()
    def generate_traces_batched(
        self,
        model_id: str,
        questions: List[str],
        num_traces: int,
        context: str = None,
    ) -> List[List[str]]:
        """
        Generate traces for multiple questions in batched mode.

        This is the key optimization - generates all questions at once.
        Uses per-model generation config for model-specific temperature, top_k, etc.
        """
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        model.eval()

        # Get per-model generation config (merges global defaults with model overrides)
        gen_config = self.get_generation_config(model_id=model_id)

        # Prepare all prompts using appropriate formatter
        prompts = []
        for question in questions:
            prompt = self._format_prompt_for_model(model_id, question, context)
            prompts.append(prompt)

        # Expand for multiple traces
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_traces)

        # Tokenize all at once
        inputs = tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_length,
        ).to(model.device)

        # Generate all traces at once using per-model config
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            top_k=gen_config["top_k"],
            do_sample=gen_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
        )

        # Decode and organize by question
        all_traces = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            trace = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )
            all_traces.append(trace)

        # Reorganize: list of lists (one list per question)
        traces_per_question = []
        for i in range(len(questions)):
            q_traces = []
            for j in range(num_traces):
                idx = i * num_traces + j
                q_traces.append(all_traces[idx])
            traces_per_question.append(q_traces)

        model.train()
        return traces_per_question

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
    ):
        """Main training loop with batched generation."""
        # Load models
        self.load_all_models()
        self.setup_optimizers()

        # Calculate total steps
        num_training_steps = (
            len(train_dataloader) * self.num_epochs
            // self.gradient_accumulation_steps
        )
        self.setup_schedulers(num_training_steps)

        logger.info(f"Starting FAST training for {self.num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Models: {list(self.models.keys())}")
        logger.info(f"Generation batch size: {self.gen_batch_size}")
        logger.info(f"torch.compile enabled: {self.use_compile}")

        for epoch in range(1, self.num_epochs + 1):
            self.state.epoch = epoch
            self.reward_fn.set_epoch(epoch)

            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")

            # Training epoch with batched generation
            self.train_epoch_batched(train_dataloader, epoch)

            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate_batched(eval_dataloader)
                logger.info(f"Eval metrics: {eval_metrics}")

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}")

            # Epoch 2: Add distillation from buddy buffer (optional)
            enable_distillation = self.config.get("collaboration", {}).get("enable_distillation", True)
            if enable_distillation and epoch == 2 and len(self.buddy_buffer) > 0:
                logger.info("Running distillation from buddy buffer...")
                self.distillation_epoch(train_dataloader)
            elif not enable_distillation and epoch == 2:
                logger.info("Distillation disabled via config (enable_distillation: false)")

        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training complete!")

    def train_epoch_batched(
        self,
        dataloader: DataLoader,
        epoch: int,
    ):
        """Train for one epoch with batched generation."""
        for model in self.models.values():
            model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_rescues = 0
        num_rescue_attempts = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        # Collect questions for batched processing
        batch_questions = []
        batch_ground_truths = []

        for step, batch in enumerate(progress_bar):
            questions = batch["question"]
            ground_truths = batch["answer"]

            # Add to batch
            batch_questions.extend(questions)
            batch_ground_truths.extend(ground_truths)

            # Process when batch is full
            if len(batch_questions) >= self.gen_batch_size:
                metrics = self.train_batch(
                    questions=batch_questions[:self.gen_batch_size],
                    ground_truths=batch_ground_truths[:self.gen_batch_size],
                    step=self.state.global_step,
                )

                # Update metrics
                epoch_loss += metrics.get("loss", 0) * self.gen_batch_size
                epoch_accuracy += metrics.get("accuracy", 0) * self.gen_batch_size
                num_rescue_attempts += metrics.get("rescue_attempts", 0)
                num_rescues += metrics.get("rescue_successes", 0)

                self.state.global_step += self.gen_batch_size

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics.get('loss', 0):.4f}",
                    "acc": f"{metrics.get('accuracy', 0):.2%}",
                    "rescue": f"{num_rescues}/{num_rescue_attempts}",
                })

                # Remove processed questions
                batch_questions = batch_questions[self.gen_batch_size:]
                batch_ground_truths = batch_ground_truths[self.gen_batch_size:]

                # Periodic checkpoint
                if self.state.global_step % self.config["training"]["save_steps"] == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

        # Process remaining questions
        if batch_questions:
            metrics = self.train_batch(
                questions=batch_questions,
                ground_truths=batch_ground_truths,
                step=self.state.global_step,
            )
            epoch_loss += metrics.get("loss", 0) * len(batch_questions)
            epoch_accuracy += metrics.get("accuracy", 0) * len(batch_questions)
            self.state.global_step += len(batch_questions)

        # Epoch summary
        num_samples = len(dataloader.dataset)
        rescue_rate = num_rescues / max(1, num_rescue_attempts)

        logger.info(f"Epoch {epoch} summary:")
        logger.info(f"  Average loss: {epoch_loss / num_samples:.4f}")
        logger.info(f"  Average accuracy: {epoch_accuracy / num_samples:.2%}")
        logger.info(f"  Rescue rate: {rescue_rate:.2%}")

    def train_batch(
        self,
        questions: List[str],
        ground_truths: List[str],
        step: int,
    ) -> Dict[str, float]:
        """Train on a batch using batched generation."""
        import sys
        logger.info(f"[Step {step}] Starting train_batch with {len(questions)} questions")
        sys.stdout.flush()
        sys.stderr.flush()

        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "rescue_attempts": 0,
            "rescue_successes": 0,
        }

        batch_size = len(questions)

        # ========== Micro-round A: Batched Cold Generation ==========
        logger.info(f"[Step {step}] Starting Round A generation")
        round_a_results = {}
        best_correct_traces = [None] * batch_size
        best_correct_models = [None] * batch_size
        any_correct = [False] * batch_size

        for model_id in self.models:
            # Generate K traces for ALL questions at once
            all_traces = self.generate_traces_batched(
                model_id=model_id,
                questions=questions,
                num_traces=self.K,
            )

            round_a_results[model_id] = []
            for q_idx, (question, gt, traces) in enumerate(zip(questions, ground_truths, all_traces)):
                round_a = self.micro_round.process_round_a(
                    traces=traces,
                    ground_truth=gt,
                    question=question,
                    reward_fn=self.reward_fn,
                )
                round_a_results[model_id].append(round_a)

                if round_a.best_correct_trace is not None:
                    any_correct[q_idx] = True
                    if best_correct_traces[q_idx] is None:
                        best_correct_traces[q_idx] = round_a.best_correct_trace
                        best_correct_models[q_idx] = model_id

        if self.log_round_rewards:
            for model_id in self.models:
                for q_idx, round_a in enumerate(round_a_results[model_id]):
                    if not round_a.rewards:
                        continue
                    rewards = np.asarray(round_a.rewards, dtype=np.float32)
                    correct = sum(round_a.is_correct)
                    best_idx = round_a.best_correct_idx
                    best_reward = None
                    if best_idx is not None and 0 <= best_idx < len(round_a.rewards):
                        best_reward = round_a.rewards[best_idx]
                    best_info = ""
                    if best_reward is not None:
                        best_info = f", best_idx={best_idx}, best_reward={best_reward:.4f}"
                    reward_list = [round(float(r), 4) for r in round_a.rewards]
                    logger.info(
                        f"[Step {step}] {model_id} Q{q_idx} Round A rewards: "
                        f"min={rewards.min():.4f}, mean={rewards.mean():.4f}, "
                        f"max={rewards.max():.4f}, correct={correct}/{len(round_a.is_correct)}{best_info}, "
                        f"values={reward_list}"
                    )
                    if self.log_round_reward_components and round_a.reward_details:
                        formatted = []
                        for detail in round_a.reward_details:
                            formatted.append({
                                k: (
                                    round(float(v), 4)
                                    if isinstance(v, (float, int, np.floating, np.integer))
                                    else v
                                )
                                for k, v in detail.items()
                            })
                        logger.info(
                            f"[Step {step}] {model_id} Q{q_idx} Round A reward details: "
                            f"{json.dumps(formatted, sort_keys=True)}"
                        )

        # ========== Micro-round B: Batched Contexted Generation ==========
        logger.info(f"[Step {step}] Round A complete. Starting Round B generation")
        round_b_results = {model_id: [None] * batch_size for model_id in self.models}

        # Prepare contexted questions
        contexted_questions = []
        contexted_indices = []
        teacher_contexts = []

        for q_idx, (question, best_trace) in enumerate(zip(questions, best_correct_traces)):
            if best_trace is not None:
                context = self.micro_round.build_teacher_context(best_trace)
                contexted_questions.append(question)
                contexted_indices.append(q_idx)
                teacher_contexts.append(context)

        if contexted_questions:
            for model_id in self.models:
                # Generate contexted traces for each question with hint dropout
                for i, (q_idx, question, context) in enumerate(
                    zip(contexted_indices, contexted_questions, teacher_contexts)
                ):
                    # Apply hint dropout like the original trainer
                    traces_b = []
                    used_hint = []

                    for _ in range(self.K_prime):
                        use_hint = self.micro_round.should_use_hint()
                        used_hint.append(use_hint)

                        if use_hint:
                            # Generate with context
                            trace_list = self.generate_traces_batched(
                                model_id=model_id,
                                questions=[question],
                                num_traces=1,
                                context=context,
                            )[0]
                        else:
                            # Generate without context (hint dropout)
                            trace_list = self.generate_traces_batched(
                                model_id=model_id,
                                questions=[question],
                                num_traces=1,
                                context=None,
                            )[0]
                        traces_b.append(trace_list[0] if trace_list else "")

                    gt = ground_truths[q_idx]
                    round_a_had_correct = round_a_results[model_id][q_idx].best_correct_trace is not None

                    round_b = self.micro_round.process_round_b(
                        traces=traces_b,
                        ground_truth=gt,
                        question=question,
                        reward_fn=self.reward_fn,
                        used_hint=used_hint,
                        round_a_had_correct=round_a_had_correct,
                    )
                    round_b_results[model_id][q_idx] = round_b

                    if round_b.rescue_success:
                        metrics["rescue_successes"] += 1
                        self.buddy_buffer.add(
                            question=question,
                            teacher_context=context,
                            rescued_model=model_id,
                            source_model=best_correct_models[q_idx],
                            original_trace=round_a_results[model_id][q_idx].traces[0] if round_a_results[model_id][q_idx].traces else "",
                            rescue_trace=traces_b[0] if traces_b else "",
                            step=step,
                        )

        if self.log_round_rewards:
            for model_id in self.models:
                for q_idx, round_b in enumerate(round_b_results[model_id]):
                    if round_b is None or not round_b.rewards:
                        continue
                    rewards = np.asarray(round_b.rewards, dtype=np.float32)
                    correct = sum(round_b.is_correct)
                    hints_used = sum(1 for used in round_b.used_hint if used)
                    reward_list = [round(float(r), 4) for r in round_b.rewards]
                    logger.info(
                        f"[Step {step}] {model_id} Q{q_idx} Round B rewards: "
                        f"min={rewards.min():.4f}, mean={rewards.mean():.4f}, "
                        f"max={rewards.max():.4f}, correct={correct}/{len(round_b.is_correct)}, "
                        f"hints_used={hints_used}/{len(round_b.used_hint)}, "
                        f"rescue_success={round_b.rescue_success}, values={reward_list}"
                    )
                    if self.log_round_reward_components and round_b.reward_details:
                        formatted = []
                        for detail in round_b.reward_details:
                            formatted.append({
                                k: (
                                    round(float(v), 4)
                                    if isinstance(v, (float, int, np.floating, np.integer))
                                    else v
                                )
                                for k, v in detail.items()
                            })
                        logger.info(
                            f"[Step {step}] {model_id} Q{q_idx} Round B reward details: "
                            f"{json.dumps(formatted, sort_keys=True)}"
                        )

        metrics["rescue_attempts"] = sum(1 for bc in best_correct_traces if bc is not None)
        logger.info(f"[Step {step}] Round B complete. Starting policy update")

        # ========== GRPO/GSPO/SAPO Update ==========
        total_loss = 0.0
        total_accuracy = 0.0

        for model_id in self.models:
            model = self.models[model_id]
            ref_model = self.ref_models[model_id]
            tokenizer = self.tokenizers[model_id]

            # Collect all traces and rewards
            all_prompts = []
            all_traces = []
            all_rewards = []
            all_sources = []
            all_q_any_correct = []  # Track if round A had any correct for each trace

            for q_idx in range(batch_size):
                question = questions[q_idx]
                round_a = round_a_results[model_id][q_idx]
                round_b = round_b_results[model_id][q_idx]

                traces, rewards, sources = self.micro_round.combine_rounds(
                    round_a, round_b, lambda_b=0.8
                )

                for trace, reward, source in zip(traces, rewards, sources):
                    prompt = format_prompt(question)
                    all_prompts.append(prompt)
                    all_traces.append(trace)
                    all_rewards.append(reward)
                    all_sources.append(source)
                    # Track if this question had any correct in round A (across all models)
                    all_q_any_correct.append(any_correct[q_idx])

            if not all_traces:
                continue

            # Normalize advantages across all traces first
            rewards_tensor = torch.tensor(all_rewards, device=model.device)
            advantages = self.loss_fn.normalize_advantages(rewards_tensor)

            # Compute is_rescue flags
            is_rescue_flags = [
                s == 'contexted' and not q_any_correct
                for s, q_any_correct in zip(all_sources, all_q_any_correct)
            ]

            # Process each trace individually and accumulate loss
            # This avoids concatenation issues with different sequence lengths
            accumulated_loss = 0.0
            num_traces = len(all_prompts)

            for i in range(num_traces):
                # Single prompt-trace pair
                log_probs, mask = self.compute_log_probs(
                    model, tokenizer, [all_prompts[i]], [all_traces[i]]
                )

                with torch.no_grad():
                    old_log_probs, _ = self.compute_log_probs(
                        model, tokenizer, [all_prompts[i]], [all_traces[i]]
                    )
                    ref_log_probs, _ = self.compute_log_probs(
                        ref_model, tokenizer, [all_prompts[i]], [all_traces[i]]
                    )

                # Single trace tensors
                adv = advantages[i:i+1]
                is_rescue = torch.tensor([is_rescue_flags[i]], device=log_probs.device)

                # Compute loss for this single trace
                loss_output = self.loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    ref_log_probs=ref_log_probs,
                    advantages=adv,
                    mask=mask,
                    is_rescue=is_rescue,
                )

                # Accumulate gradients
                trace_loss = loss_output.loss / (num_traces * self.gradient_accumulation_steps)
                trace_loss.backward()
                accumulated_loss += trace_loss.item()

            total_loss += accumulated_loss

            # Accuracy
            model_accuracy = sum(
                1 for r in round_a_results[model_id]
                if r.best_correct_trace is not None
            ) / batch_size
            total_accuracy += model_accuracy

            # Gradient step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizers[model_id].step()
                self.schedulers[model_id].step()
                self.optimizers[model_id].zero_grad()

        num_models = len(self.models)
        metrics["loss"] = total_loss / num_models
        metrics["accuracy"] = total_accuracy / num_models
        logger.info(f"[Step {step}] Batch complete. Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2%}")

        return metrics

    def evaluate_batched(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate with batched generation."""
        for model in self.models.values():
            model.eval()

        metrics = {model_id: {"correct": 0, "total": 0} for model_id in self.models}

        # Collect all questions
        all_questions = []
        all_ground_truths = []
        for batch in dataloader:
            all_questions.extend(batch["question"])
            all_ground_truths.extend(batch["answer"])

        # Process in batches
        for i in range(0, len(all_questions), self.gen_batch_size):
            batch_q = all_questions[i:i+self.gen_batch_size]
            batch_gt = all_ground_truths[i:i+self.gen_batch_size]

            for model_id in self.models:
                traces = self.generate_traces_batched(
                    model_id=model_id,
                    questions=batch_q,
                    num_traces=1,
                )

                for j, (question, gt, q_traces) in enumerate(zip(batch_q, batch_gt, traces)):
                    result = self.reward_fn.exploit_reward(q_traces[0], gt, question)
                    metrics[model_id]["total"] += 1
                    if result.is_correct:
                        metrics[model_id]["correct"] += 1

        # Compute final metrics
        eval_metrics = {}
        for model_id, m in metrics.items():
            acc = m["correct"] / max(1, m["total"])
            eval_metrics[f"{model_id}_accuracy"] = acc

        for model in self.models.values():
            model.train()

        return eval_metrics

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
                    traces = self.generate_traces_batched(
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
