"""
Distributed Collaborative Trainer
=================================

Multi-GPU training using Accelerate/DeepSpeed.

Key features:
1. Data parallel training across multiple GPUs
2. DeepSpeed ZeRO Stage 2 for memory efficiency
3. Gradient accumulation handled by Accelerator
4. Synchronized generation across processes

Usage:
    accelerate launch --num_processes 4 scripts/train_distributed.py --config config.yaml
"""

import os
import random

# Ensure HuggingFace cache is set correctly before any imports
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .base_trainer import BaseCollabTrainer, TrainingState
from .micro_rounds import MicroRoundManager
from .buddy_buffer import BuddyBuffer
from ..losses import get_loss_fn
from ..rewards import CombinedRewardFunction
from ..data.preprocessing import format_prompt


logger = logging.getLogger(__name__)


class DistributedCollaborativeTrainer(BaseCollabTrainer):
    """
    Multi-GPU collaborative trainer using Accelerate/DeepSpeed.

    Strategy:
    - Both M1 and M2 are wrapped with the same Accelerator
    - Data parallel: same model replicated across GPUs
    - Generation synchronized across ranks
    - Gradient accumulation handled by Accelerator
    """

    def __init__(
        self,
        config: dict,
        model_configs: List[dict],
        output_dir: str,
        deepspeed_config: str = None,
    ):
        # Initialize Accelerator BEFORE parent class
        deepspeed_plugin = None
        if deepspeed_config:
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=deepspeed_config,
                gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            mixed_precision="bf16" if config["training"]["bf16"] else "fp16",
            deepspeed_plugin=deepspeed_plugin,
            log_with="wandb" if config.get("use_wandb") else None,
        )

        # Set seed for reproducibility
        set_seed(config.get("seed", 42))

        # Store distributed info
        self.is_main_process = self.accelerator.is_main_process
        self.num_processes = self.accelerator.num_processes
        self.local_rank = self.accelerator.local_process_index

        # Initialize parent class (will use our overridden load_model)
        super().__init__(config, model_configs, output_dir)

        # Micro-round manager
        self.micro_round = MicroRoundManager(config)

        # Buddy buffer
        self.buddy_buffer = BuddyBuffer(
            max_size=config["collaboration"]["max_buffer_size"]
        )

        # Generation batch size
        self.gen_batch_size = config.get("fast_training", {}).get("gen_batch_size", 4)

        # Distillation settings
        self.distillation_batch_size = config["collaboration"]["distillation_batch_size"]
        self.distillation_frequency = config["collaboration"]["distillation_frequency"]

        if self.is_main_process:
            logger.info(f"Initialized distributed trainer with {self.num_processes} GPUs")
            logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
            logger.info(f"DeepSpeed enabled: {deepspeed_config is not None}")

    def load_model(
        self,
        model_name: str,
        model_id: str,
        use_lora: bool = True,
    ):
        """
        Load model WITHOUT device_map for Accelerate compatibility.

        Note: Accelerator handles device placement, so we don't use device_map="auto"
        """
        if self.is_main_process:
            logger.info(f"Loading model {model_id}: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # CRITICAL: Do NOT use device_map="auto" with Accelerate/DeepSpeed
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
            trust_remote_code=True,
            # NO device_map - Accelerate handles this
        )

        if use_lora and self.config["training"]["use_lora"]:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["training"]["lora_r"],
                lora_alpha=self.config["training"]["lora_alpha"],
                lora_dropout=self.config["training"]["lora_dropout"],
                target_modules=self.config["training"]["lora_target_modules"],
            )
            model = get_peft_model(model, lora_config)
            if self.is_main_process:
                model.print_trainable_parameters()

        return model, tokenizer

    def load_all_models(self):
        """Load all models for distributed training."""
        for i, model_config in enumerate(self.model_configs):
            model_id = f"M{i+1}"
            model_name = model_config["name"]

            model, tokenizer = self.load_model(model_name, model_id)
            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer

            # Reference model (NOT prepared with Accelerator - stays frozen on single device)
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
                trust_remote_code=True,
            )
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            # Move reference model to accelerator device
            ref_model = ref_model.to(self.accelerator.device)
            self.ref_models[model_id] = ref_model

    def setup_optimizers(self):
        """Setup optimizers for each model."""
        for model_id, model in self.models.items():
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config["training"]["weight_decay"],
            )
            self.optimizers[model_id] = optimizer

    def prepare_for_training(self, train_dataloader: DataLoader):
        """Prepare all components with Accelerator."""
        # Prepare models, optimizers together
        prepared_dataloader = None

        for model_id in self.models:
            model, optimizer, dataloader = self.accelerator.prepare(
                self.models[model_id],
                self.optimizers[model_id],
                train_dataloader,
            )
            self.models[model_id] = model
            self.optimizers[model_id] = optimizer
            prepared_dataloader = dataloader  # Same for all models

        return prepared_dataloader

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
    ):
        """Main training loop with distributed support."""
        # Load models
        self.load_all_models()
        self.setup_optimizers()

        # Prepare with Accelerator
        train_dataloader = self.prepare_for_training(train_dataloader)

        # Calculate total steps
        num_training_steps = (
            len(train_dataloader) * self.num_epochs
            // self.gradient_accumulation_steps
        )
        self.setup_schedulers(num_training_steps)

        # Prepare schedulers
        for model_id in self.schedulers:
            self.schedulers[model_id] = self.accelerator.prepare(
                self.schedulers[model_id]
            )

        if self.is_main_process:
            logger.info(f"Starting DISTRIBUTED training for {self.num_epochs} epochs")
            logger.info(f"Total training steps: {num_training_steps}")
            logger.info(f"Number of GPUs: {self.num_processes}")
            logger.info(f"Per-GPU batch size: {self.batch_size}")
            logger.info(f"Effective batch size: {self.batch_size * self.num_processes * self.gradient_accumulation_steps}")

        for epoch in range(1, self.num_epochs + 1):
            self.state.epoch = epoch
            self.reward_fn.set_epoch(epoch)

            if self.is_main_process:
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch {epoch}/{self.num_epochs}")
                logger.info(f"{'='*50}")

            self.train_epoch_distributed(train_dataloader, epoch)

            # Evaluation on main process only
            if eval_dataloader is not None and self.is_main_process:
                eval_metrics = self.evaluate_distributed(eval_dataloader)
                logger.info(f"Eval metrics: {eval_metrics}")

            # Wait for all processes
            self.accelerator.wait_for_everyone()

            # Save checkpoint (main process only)
            if self.is_main_process:
                self.save_checkpoint(f"epoch_{epoch}")

            # Epoch 2: Distillation
            if epoch == 2 and len(self.buddy_buffer) > 0:
                if self.is_main_process:
                    logger.info("Running distillation from buddy buffer...")
                self.distillation_epoch(train_dataloader)

        # Wait and save final
        self.accelerator.wait_for_everyone()
        if self.is_main_process:
            self.save_checkpoint("final")
            logger.info("Training complete!")

    def train_epoch_distributed(
        self,
        dataloader: DataLoader,
        epoch: int,
    ):
        """Train for one epoch with distributed data parallel."""
        for model in self.models.values():
            model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_rescues = 0
        num_rescue_attempts = 0
        num_samples = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.is_main_process,
        )

        batch_questions = []
        batch_ground_truths = []

        for step, batch in enumerate(progress_bar):
            questions = batch["question"]
            ground_truths = batch["answer"]

            batch_questions.extend(questions)
            batch_ground_truths.extend(ground_truths)

            if len(batch_questions) >= self.gen_batch_size:
                # Use accumulation context
                with self.accelerator.accumulate(*self.models.values()):
                    metrics = self.train_batch_distributed(
                        questions=batch_questions[:self.gen_batch_size],
                        ground_truths=batch_ground_truths[:self.gen_batch_size],
                        step=self.state.global_step,
                    )

                # Update metrics
                batch_size = self.gen_batch_size
                epoch_loss += metrics.get("loss", 0) * batch_size
                epoch_accuracy += metrics.get("accuracy", 0) * batch_size
                num_rescue_attempts += metrics.get("rescue_attempts", 0)
                num_rescues += metrics.get("rescue_successes", 0)
                num_samples += batch_size

                self.state.global_step += batch_size

                if self.is_main_process:
                    progress_bar.set_postfix({
                        "loss": f"{metrics.get('loss', 0):.4f}",
                        "acc": f"{metrics.get('accuracy', 0):.2%}",
                        "rescue": f"{num_rescues}/{num_rescue_attempts}",
                    })

                batch_questions = batch_questions[self.gen_batch_size:]
                batch_ground_truths = batch_ground_truths[self.gen_batch_size:]

                # Periodic checkpoint
                if self.is_main_process and self.state.global_step % self.config["training"]["save_steps"] == 0:
                    self.save_checkpoint(f"step_{self.state.global_step}")

        # Process remaining
        if batch_questions:
            with self.accelerator.accumulate(*self.models.values()):
                metrics = self.train_batch_distributed(
                    questions=batch_questions,
                    ground_truths=batch_ground_truths,
                    step=self.state.global_step,
                )
            epoch_loss += metrics.get("loss", 0) * len(batch_questions)
            epoch_accuracy += metrics.get("accuracy", 0) * len(batch_questions)
            num_samples += len(batch_questions)
            self.state.global_step += len(batch_questions)

        # Epoch summary
        if self.is_main_process and num_samples > 0:
            rescue_rate = num_rescues / max(1, num_rescue_attempts)
            logger.info(f"Epoch {epoch} summary:")
            logger.info(f"  Average loss: {epoch_loss / num_samples:.4f}")
            logger.info(f"  Average accuracy: {epoch_accuracy / num_samples:.2%}")
            logger.info(f"  Rescue rate: {rescue_rate:.2%}")

    @torch.no_grad()
    def generate_traces_distributed(
        self,
        model_id: str,
        questions: List[str],
        num_traces: int,
        context: str = None,
    ) -> List[List[str]]:
        """
        Generate traces with distributed synchronization.

        Uses accelerator.unwrap_model for generation.
        """
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]

        # Unwrap model for generation (DeepSpeed wraps models)
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.eval()

        # Prepare prompts
        prompts = []
        for question in questions:
            if context:
                prompt = f"{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"
            else:
                prompt = format_prompt(question)
            prompts.append(prompt)

        # Expand for multiple traces
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_traces)

        # Tokenize
        inputs = tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.accelerator.device)

        # Generate with synced_gpus for DeepSpeed
        outputs = unwrapped_model.generate(
            **inputs,
            max_new_tokens=self.gen_config["max_new_tokens"],
            temperature=self.gen_config["temperature"],
            top_p=self.gen_config["top_p"],
            top_k=self.gen_config["top_k"],
            do_sample=self.gen_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            synced_gpus=self.accelerator.distributed_type != "NO",
        )

        # Decode
        all_traces = []
        for output in outputs:
            input_len = inputs["input_ids"].shape[1]
            trace = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            all_traces.append(trace)

        # Reorganize by question
        traces_per_question = []
        for i in range(len(questions)):
            q_traces = [all_traces[i * num_traces + j] for j in range(num_traces)]
            traces_per_question.append(q_traces)

        unwrapped_model.train()
        return traces_per_question

    def train_batch_distributed(
        self,
        questions: List[str],
        ground_truths: List[str],
        step: int,
    ) -> Dict[str, float]:
        """Train on a batch with distributed gradient sync."""
        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "rescue_attempts": 0,
            "rescue_successes": 0,
        }

        batch_size = len(questions)

        # ========== Micro-round A: Cold Generation ==========
        round_a_results = {}
        best_correct_traces = [None] * batch_size
        best_correct_models = [None] * batch_size
        any_correct = [False] * batch_size

        for model_id in self.models:
            all_traces = self.generate_traces_distributed(
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

        # ========== Micro-round B: Contexted Generation ==========
        # CRITICAL FIX: All ranks must call generate() the same number of times.
        # Different ranks may have different local data, but synced_gpus=True requires
        # identical call patterns. We iterate over ALL questions (not just those with
        # contexts) to ensure synchronized iteration counts across ranks.
        round_b_results = {model_id: [None] * batch_size for model_id in self.models}

        # Pre-compute contexts for all questions (None if no best trace available)
        teacher_contexts = []
        for q_idx, (question, best_trace) in enumerate(zip(questions, best_correct_traces)):
            if best_trace is not None:
                context = self.micro_round.compress_trace(
                    best_trace,
                    include_answer=self.config["collaboration"]["include_answer_in_context"],
                )
                teacher_contexts.append(context)
            else:
                teacher_contexts.append(None)

        # Pre-compute ALL hint decisions with synced random seed across ranks
        # This ensures all ranks make identical decisions and follow same code path
        sync_seed = step + self.state.epoch * 10000  # Deterministic seed based on step
        hint_rng = random.Random(sync_seed)
        p_hint = self.config["collaboration"]["p_hint"]

        # Pre-compute hint decisions for all (model, question, k_prime) combinations
        # Structure: hint_decisions[model_id][q_idx][k] = bool
        hint_decisions = {}
        for model_id in self.models:
            hint_decisions[model_id] = []
            for _ in range(batch_size):  # Iterate over ALL questions, not just contexted
                k_hints = [hint_rng.random() < p_hint for _ in range(self.K_prime)]
                hint_decisions[model_id].append(k_hints)

        # Synchronize all processes before generation to ensure same state
        self.accelerator.wait_for_everyone()

        # Iterate over ALL questions to ensure same generate() call count across ranks
        for model_id in self.models:
            for q_idx, (question, context, gt) in enumerate(
                zip(questions, teacher_contexts, ground_truths)
            ):
                traces_b = []
                used_hint = hint_decisions[model_id][q_idx]

                for k in range(self.K_prime):
                    use_hint = used_hint[k]

                    if use_hint and context is not None:
                        trace_list = self.generate_traces_distributed(
                            model_id=model_id,
                            questions=[question],
                            num_traces=1,
                            context=context,
                        )[0]
                    else:
                        # Generate without context (for both no-hint case and no-context case)
                        trace_list = self.generate_traces_distributed(
                            model_id=model_id,
                            questions=[question],
                            num_traces=1,
                            context=None,
                        )[0]
                    traces_b.append(trace_list[0] if trace_list else "")

                # Only process Round B results if we had a context (rescue attempt)
                if context is not None:
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

        metrics["rescue_attempts"] = sum(1 for bc in best_correct_traces if bc is not None)

        # ========== Policy Update ==========
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
            all_q_any_correct = []

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
                    all_q_any_correct.append(any_correct[q_idx])

            if not all_traces:
                continue

            # Normalize advantages
            rewards_tensor = torch.tensor(all_rewards, device=self.accelerator.device)
            advantages = self.loss_fn.normalize_advantages(rewards_tensor)

            # Compute is_rescue flags
            is_rescue_flags = [
                s == 'contexted' and not q_any_correct
                for s, q_any_correct in zip(all_sources, all_q_any_correct)
            ]

            # Process each trace
            accumulated_loss = 0.0
            num_traces = len(all_prompts)

            for i in range(num_traces):
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

                adv = advantages[i:i+1]
                is_rescue = torch.tensor([is_rescue_flags[i]], device=self.accelerator.device)

                loss_output = self.loss_fn(
                    log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    ref_log_probs=ref_log_probs,
                    advantages=adv,
                    mask=mask,
                    is_rescue=is_rescue,
                )

                # Use accelerator.backward
                trace_loss = loss_output.loss / num_traces
                self.accelerator.backward(trace_loss)
                accumulated_loss += trace_loss.item()

            total_loss += accumulated_loss

            # Accuracy
            model_accuracy = sum(
                1 for r in round_a_results[model_id]
                if r.best_correct_trace is not None
            ) / batch_size
            total_accuracy += model_accuracy

            # Optimizer step (Accelerator handles sync)
            self.optimizers[model_id].step()
            self.schedulers[model_id].step()
            self.optimizers[model_id].zero_grad()

        num_models = len(self.models)
        metrics["loss"] = total_loss / num_models
        metrics["accuracy"] = total_accuracy / num_models

        return metrics

    def evaluate_distributed(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate with distributed generation."""
        for model in self.models.values():
            model.eval()

        metrics = {model_id: {"correct": 0, "total": 0} for model_id in self.models}

        all_questions = []
        all_ground_truths = []
        for batch in dataloader:
            all_questions.extend(batch["question"])
            all_ground_truths.extend(batch["answer"])

        for i in range(0, len(all_questions), self.gen_batch_size):
            batch_q = all_questions[i:i+self.gen_batch_size]
            batch_gt = all_ground_truths[i:i+self.gen_batch_size]

            for model_id in self.models:
                traces = self.generate_traces_distributed(
                    model_id=model_id,
                    questions=batch_q,
                    num_traces=1,
                )

                for j, (question, gt, q_traces) in enumerate(zip(batch_q, batch_gt, traces)):
                    result = self.reward_fn.exploit_reward(q_traces[0], gt, question)
                    metrics[model_id]["total"] += 1
                    if result.is_correct:
                        metrics[model_id]["correct"] += 1

        eval_metrics = {}
        for model_id, m in metrics.items():
            acc = m["correct"] / max(1, m["total"])
            eval_metrics[f"{model_id}_accuracy"] = acc

        for model in self.models.values():
            model.train()

        return eval_metrics

    def distillation_epoch(self, dataloader: DataLoader):
        """Run distillation from buddy buffer."""
        if self.is_main_process:
            logger.info(f"Distillation with {len(self.buddy_buffer)} buffer entries")

        samples = self.buddy_buffer.sample_for_distillation(
            batch_size=self.distillation_batch_size * 10
        )

        for model_id, entries in samples.items():
            if not entries:
                continue

            model = self.models[model_id]
            tokenizer = self.tokenizers[model_id]

            if self.is_main_process:
                logger.info(f"Distillation for {model_id}: {len(entries)} samples")

            for entry in tqdm(entries, desc=f"Distill {model_id}", disable=not self.is_main_process):
                prompt = self.micro_round.format_contexted_prompt(
                    entry.question,
                    entry.teacher_context,
                )

                target_trace = entry.rescue_trace if entry.rescue_trace else None

                if target_trace is None:
                    traces = self.generate_traces_distributed(
                        model_id=model_id,
                        questions=[entry.question],
                        num_traces=1,
                        context=entry.teacher_context,
                    )
                    target_trace = traces[0][0] if traces and traces[0] else None

                if target_trace is None:
                    continue

                inputs = tokenizer(
                    prompt + target_trace,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.accelerator.device)

                # Unwrap for forward pass if needed
                unwrapped = self.accelerator.unwrap_model(model)
                outputs = unwrapped(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / self.gradient_accumulation_steps

                self.accelerator.backward(loss)
                self.optimizers[model_id].step()
                self.optimizers[model_id].zero_grad()

    def save_checkpoint(self, tag: str = "latest"):
        """Save checkpoint with Accelerator (handles distributed state)."""
        if not self.is_main_process:
            return

        checkpoint_dir = Path(self.output_dir) / f"checkpoint-{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        # Wait for all processes
        self.accelerator.wait_for_everyone()

        # Save each model's state
        for model_id, model in self.models.items():
            model_dir = checkpoint_dir / model_id
            model_dir.mkdir(exist_ok=True)

            # Unwrap and save
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                model_dir,
                save_function=self.accelerator.save,
            )
            self.tokenizers[model_id].save_pretrained(model_dir)

        # Save training state
        state_dict = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_accuracy": self.state.best_accuracy,
            "best_rescue_rate": self.state.best_rescue_rate,
            "metrics_history": self.state.metrics_history,
        }

        import json
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        # Save buddy buffer
        self.buddy_buffer.save(str(checkpoint_dir / "buddy_buffer.json"))

        logger.info(f"Checkpoint saved to {checkpoint_dir}")
