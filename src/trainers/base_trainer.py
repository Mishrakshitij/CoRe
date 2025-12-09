"""
Base Collaborative Trainer
==========================

Foundation class for collaborative training with GRPO/GSPO/SAPO.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

from ..losses import get_loss_fn
from ..rewards import CombinedRewardFunction


logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training progress."""
    epoch: int = 0
    global_step: int = 0
    best_accuracy: float = 0.0
    best_rescue_rate: float = 0.0
    training_loss: float = 0.0
    metrics_history: List[Dict] = field(default_factory=list)


class BaseCollabTrainer:
    """
    Base trainer for collaborative reasoning.

    Handles:
    - Model loading with LoRA
    - Optimizer and scheduler setup
    - Basic training loop structure
    - Checkpointing and logging
    """

    def __init__(
        self,
        config: dict,
        model_configs: List[dict],
        output_dir: str,
    ):
        self.config = config
        self.model_configs = model_configs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        self.num_epochs = config["training"]["num_epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.learning_rate = config["training"]["learning_rate"]
        self.max_grad_norm = config["training"]["max_grad_norm"]

        # Policy optimization config
        self.po_config = config["policy_optimization"]
        self.K = self.po_config["K"]
        self.K_prime = self.po_config["K_prime"]

        # Generation config
        self.gen_config = config["generation"]

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.models = {}
        self.tokenizers = {}
        self.optimizers = {}
        self.schedulers = {}
        self.ref_models = {}  # Reference models for KL

        # Loss function
        self.loss_fn = get_loss_fn(
            self.po_config["algorithm"],
            self.po_config,
        )

        # Reward function
        self.reward_fn = CombinedRewardFunction(config["rewards"])

        # Training state
        self.state = TrainingState()

        # Logging
        self.use_wandb = config.get("use_wandb", False)

    def load_model(
        self,
        model_name: str,
        model_id: str,
        use_lora: bool = True,
    ) -> Tuple[nn.Module, Any]:
        """
        Load a model with optional LoRA.

        Args:
            model_name: HuggingFace model name
            model_id: Identifier for this model (e.g., 'M1', 'M2')
            use_lora: Whether to apply LoRA

        Returns:
            (model, tokenizer)
        """
        logger.info(f"Loading model {model_id}: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Apply LoRA if enabled
        if use_lora and self.config["training"]["use_lora"]:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["training"]["lora_r"],
                lora_alpha=self.config["training"]["lora_alpha"],
                lora_dropout=self.config["training"]["lora_dropout"],
                target_modules=self.config["training"]["lora_target_modules"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

    def load_all_models(self):
        """Load all models specified in config."""
        for i, model_config in enumerate(self.model_configs):
            model_id = f"M{i+1}"
            model_name = model_config["name"]

            model, tokenizer = self.load_model(model_name, model_id)

            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer

            # Create reference model (frozen copy)
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
                trust_remote_code=True,
                device_map="auto",
            )
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            self.ref_models[model_id] = ref_model

    def setup_optimizers(self):
        """Setup optimizers and schedulers for all models."""
        for model_id, model in self.models.items():
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config["training"]["weight_decay"],
            )
            self.optimizers[model_id] = optimizer

            # Scheduler will be set when we know total steps
            self.schedulers[model_id] = None

    def setup_schedulers(self, num_training_steps: int):
        """Setup learning rate schedulers."""
        for model_id, optimizer in self.optimizers.items():
            scheduler = get_scheduler(
                self.config["training"]["lr_scheduler"],
                optimizer=optimizer,
                num_warmup_steps=int(num_training_steps * self.config["training"]["warmup_ratio"]),
                num_training_steps=num_training_steps,
            )
            self.schedulers[model_id] = scheduler

    @torch.no_grad()
    def generate_traces(
        self,
        model_id: str,
        questions: List[str],
        num_traces: int,
        context: str = None,
    ) -> List[List[str]]:
        """
        Generate reasoning traces for questions.

        Args:
            model_id: Which model to use
            questions: List of questions
            num_traces: Number of traces per question
            context: Optional context to prepend

        Returns:
            List of trace lists (one list per question)
        """
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        model.eval()

        all_traces = []

        for question in questions:
            # Prepare prompt
            if context:
                prompt = f"{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"
            else:
                prompt = f"Question: {question}\n\nLet's solve this step by step:"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            # Generate multiple traces
            traces = []
            for _ in range(num_traces):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.gen_config["max_new_tokens"],
                    temperature=self.gen_config["temperature"],
                    top_p=self.gen_config["top_p"],
                    top_k=self.gen_config["top_k"],
                    do_sample=self.gen_config["do_sample"],
                    pad_token_id=tokenizer.pad_token_id,
                )

                trace = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                traces.append(trace)

            all_traces.append(traces)

        model.train()
        return all_traces

    def compute_log_probs(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompts: List[str],
        responses: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for responses.

        Returns:
            (log_probs, mask) both of shape [batch, seq_len]
        """
        # Tokenize prompts and responses
        full_texts = [p + r for p, r in zip(prompts, responses)]

        encodings = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        prompt_encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        # Forward pass
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )

        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = encodings["input_ids"][:, 1:].contiguous()

        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Create mask for response tokens only
        mask = torch.zeros_like(token_log_probs)
        for i, prompt_len in enumerate(prompt_encodings["attention_mask"].sum(dim=1)):
            mask[i, prompt_len-1:] = encodings["attention_mask"][i, prompt_len:]

        return token_log_probs, mask

    def save_checkpoint(self, tag: str = "latest"):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save models
        for model_id, model in self.models.items():
            model_dir = checkpoint_dir / model_id
            model.save_pretrained(model_dir)
            self.tokenizers[model_id].save_pretrained(model_dir)

        # Save training state
        state_dict = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_accuracy": self.state.best_accuracy,
            "best_rescue_rate": self.state.best_rescue_rate,
            "metrics_history": self.state.metrics_history,
        }

        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        # Save optimizer states
        for model_id, optimizer in self.optimizers.items():
            torch.save(
                optimizer.state_dict(),
                checkpoint_dir / f"optimizer_{model_id}.pt",
            )

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load training state
        with open(checkpoint_dir / "training_state.json") as f:
            state_dict = json.load(f)

        self.state.epoch = state_dict["epoch"]
        self.state.global_step = state_dict["global_step"]
        self.state.best_accuracy = state_dict["best_accuracy"]
        self.state.best_rescue_rate = state_dict["best_rescue_rate"]
        self.state.metrics_history = state_dict["metrics_history"]

        # Load models
        for model_id in self.models:
            model_dir = checkpoint_dir / model_id
            if model_dir.exists():
                # Load LoRA weights
                self.models[model_id].load_adapter(model_dir)

        # Load optimizer states
        for model_id, optimizer in self.optimizers.items():
            opt_path = checkpoint_dir / f"optimizer_{model_id}.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path))

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console and wandb."""
        # Console logging
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Step {step}: {metrics_str}")

        # Wandb logging
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Save to history
        metrics["step"] = step
        self.state.metrics_history.append(metrics)
