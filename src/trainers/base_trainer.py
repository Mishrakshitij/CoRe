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

# Mistral-3 models require transformers>=5.0.0 and use special model classes
# These imports will fail gracefully on older transformers versions
try:
    from transformers import Mistral3ForConditionalGeneration
    MISTRAL3_AVAILABLE = True
except ImportError:
    MISTRAL3_AVAILABLE = False

from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

from ..losses import get_loss_fn
from ..rewards import CombinedRewardFunction, MultiStrategyReward
from ..data.preprocessing import (
    format_prompt,
    format_multi_strategy_prompt,
    format_multi_strategy_contexted_prompt,
    extract_xml_answer,
)


logger = logging.getLogger(__name__)


def is_mistral3_model(model_name: str) -> bool:
    """
    Check if model is a Mistral-3 reasoning model that requires special handling.

    Mistral-3 models use Mistral3ForConditionalGeneration instead of AutoModelForCausalLM
    and require transformers>=5.0.0.

    Args:
        model_name: HuggingFace model name

    Returns:
        True if model is a Mistral-3 reasoning model
    """
    mistral3_patterns = [
        "Ministral-3",
        "ministral-3",
        "Mistral-3",
        "mistral-3",
    ]
    return any(pattern in model_name for pattern in mistral3_patterns)


def build_device_map_for_gpus(model_name: str, gpu_ids: List[int]) -> dict:
    """
    Build a device_map that distributes model layers across specified GPUs.

    Args:
        model_name: HuggingFace model name
        gpu_ids: List of GPU IDs to use (e.g., [0, 1] or [2, 3])

    Returns:
        device_map dict mapping model layers to specific GPUs
    """
    from transformers import AutoConfig

    logger.info(f"Building device map for {model_name} on GPUs {gpu_ids}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Handle different config structures (Mistral3 has text_config)
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
        num_layers = config.text_config.num_hidden_layers
        is_mistral3 = True
    else:
        num_layers = config.num_hidden_layers
        is_mistral3 = False

    num_gpus = len(gpu_ids)

    # Calculate layers per GPU
    layers_per_gpu = num_layers // num_gpus
    extra_layers = num_layers % num_gpus

    device_map = {}

    # Mistral3 uses model.language_model.* prefix, others use model.*
    if is_mistral3:
        device_map["model.language_model.embed_tokens"] = gpu_ids[0]
        layer_prefix = "model.language_model.layers"
        norm_key = "model.language_model.norm"
        lm_head_key = "model.lm_head"
    else:
        device_map["model.embed_tokens"] = gpu_ids[0]
        layer_prefix = "model.layers"
        norm_key = "model.norm"
        lm_head_key = "lm_head"

    # Distribute transformer layers across GPUs
    layer_idx = 0
    for i, gpu_id in enumerate(gpu_ids):
        # Add extra layers to first GPUs
        n_layers = layers_per_gpu + (1 if i < extra_layers else 0)
        for _ in range(n_layers):
            device_map[f"{layer_prefix}.{layer_idx}"] = gpu_id
            layer_idx += 1

    # Final norm and lm_head on last GPU
    device_map[norm_key] = gpu_ids[-1]
    device_map[lm_head_key] = gpu_ids[-1]

    # Rotary embedding if present (Qwen models)
    if hasattr(config, "rope_scaling") or "qwen" in model_name.lower():
        device_map["model.rotary_emb"] = gpu_ids[0]

    logger.info(f"Device map created: {len(device_map)} components across GPUs {gpu_ids}")
    logger.info(f"  Layers 0-{layers_per_gpu-1 + (1 if extra_layers > 0 else 0)} on GPU {gpu_ids[0]}")
    if num_gpus > 1:
        logger.info(f"  Layers {layer_idx - layers_per_gpu}-{layer_idx-1} on GPU {gpu_ids[-1]}")

    return device_map


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

        # Multi-strategy prompting config
        self.prompting_config = config.get("prompting", {})
        self.multi_strategy = self.prompting_config.get("multi_strategy", False)

        # Multi-strategy reward function (used when multi_strategy=True)
        if self.multi_strategy:
            logger.info("Multi-strategy prompting enabled")
            self.ms_reward_fn = MultiStrategyReward(
                w_correct=self.prompting_config.get("ms_w_correct", 2.0),
                w_diversity=self.prompting_config.get("ms_w_diversity", 0.5),
                w_consistency=self.prompting_config.get("ms_w_consistency", 1.5),
                w_format=self.prompting_config.get("ms_w_format", 0.3),
                diversity_threshold=self.prompting_config.get("diversity_threshold", 0.8),
            )
        else:
            self.ms_reward_fn = None

        # Training state
        self.state = TrainingState()

        # Logging
        self.use_wandb = config.get("use_wandb", False)

    def get_generation_config(self, model_id: str = None, model_idx: int = None) -> dict:
        """
        Get generation config for a specific model, merging global defaults with per-model overrides.

        Per-model generation config can be specified in the model's config under 'generation' key:
        ```yaml
        models:
          available:
            phi4_reasoning:
              name: "microsoft/Phi-4-reasoning-plus"
              generation:
                temperature: 0.8
                top_k: 50
        ```

        Args:
            model_id: Model identifier (e.g., 'M1', 'M2')
            model_idx: Model index (0, 1, ...) - converted to model_id if model_id not provided

        Returns:
            Merged generation config dict
        """
        # Start with global generation config
        gen_config = dict(self.gen_config)

        # Determine which model config to look up
        if model_id is None and model_idx is not None:
            model_id = f"M{model_idx + 1}"

        if model_id is None:
            return gen_config

        # Find the model config by model_id
        model_idx_from_id = int(model_id[1:]) - 1 if model_id.startswith("M") else None

        if model_idx_from_id is not None and model_idx_from_id < len(self.model_configs):
            model_config = self.model_configs[model_idx_from_id]

            # Merge per-model generation overrides if present
            if "generation" in model_config:
                per_model_gen = model_config["generation"]
                for key, value in per_model_gen.items():
                    gen_config[key] = value
                logger.debug(f"Model {model_id} using custom generation config: {per_model_gen}")

        return gen_config

    def get_effective_reward_fn(self):
        """
        Get the effective reward function based on config.

        Returns:
            MultiStrategyReward if multi_strategy is enabled, else CombinedRewardFunction
        """
        if self.multi_strategy and self.ms_reward_fn is not None:
            return self.ms_reward_fn
        return self.reward_fn

    def load_model(
        self,
        model_name: str,
        model_id: str,
        use_lora: bool = True,
    ) -> Tuple[nn.Module, Any]:
        """
        Load a model with optional LoRA/QLoRA and custom GPU assignment.

        Args:
            model_name: HuggingFace model name
            model_id: Identifier for this model (e.g., 'M1', 'M2')
            use_lora: Whether to apply LoRA

        Returns:
            (model, tokenizer)
        """
        logger.info(f"Loading model {model_id}: {model_name}")

        # Check for custom GPU mapping in config
        gpu_mapping = self.config.get("training", {}).get("model_gpu_mapping", {})
        gpu_ids = gpu_mapping.get(model_id, None)

        # Check for QLoRA settings
        use_qlora = self.config.get("training", {}).get("use_qlora", False)
        qlora_bits = self.config.get("training", {}).get("qlora_bits", 4)
        max_memory_per_gpu = self.config.get("training", {}).get("max_memory_per_gpu", None)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if this is a Mistral-3 model requiring special handling
        use_mistral3 = is_mistral3_model(model_name) and MISTRAL3_AVAILABLE

        # Build quantization config for QLoRA if enabled
        quantization_config = None
        if use_qlora:
            from transformers import BitsAndBytesConfig
            logger.info(f"Enabling QLoRA with {qlora_bits}-bit quantization for {model_id}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(qlora_bits == 4),
                load_in_8bit=(qlora_bits == 8),
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Determine device_map and max_memory based on config
        # For QLoRA + GPU mapping, use device_map="auto" with max_memory constraint
        # This lets HuggingFace handle quantized model sharding properly
        max_memory = None

        if use_mistral3:
            logger.info(f"Mistral3 model detected, using device_map='auto' for {model_id}")
            device_map = "auto"
            if gpu_ids and max_memory_per_gpu:
                max_memory = {gpu: max_memory_per_gpu for gpu in gpu_ids}
                logger.info(f"  Constraining to GPUs {gpu_ids} with max_memory={max_memory_per_gpu}")
        elif use_qlora and gpu_ids:
            # For QLoRA with specific GPUs, use auto + max_memory
            # This is more reliable than manual layer mapping for quantized models
            device_map = "auto"
            if max_memory_per_gpu:
                max_memory = {gpu: max_memory_per_gpu for gpu in gpu_ids}
            else:
                # Default: allow full GPU memory on assigned GPUs
                max_memory = {gpu: "40GB" for gpu in gpu_ids}
            logger.info(f"QLoRA + GPU mapping: using device_map='auto' with max_memory={max_memory}")
        elif gpu_ids:
            logger.info(f"Using custom GPU mapping for {model_id}: GPUs {gpu_ids}")
            device_map = build_device_map_for_gpus(model_name, gpu_ids)
        else:
            device_map = "auto"

        # Load model with appropriate settings
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        if use_mistral3:
            logger.info(f"Using Mistral3ForConditionalGeneration for {model_name}")
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                **load_kwargs,
            )
        else:
            if is_mistral3_model(model_name) and not MISTRAL3_AVAILABLE:
                logger.warning(
                    f"Mistral-3 model detected but Mistral3ForConditionalGeneration not available. "
                    f"Requires transformers>=5.0.0. Falling back to AutoModelForCausalLM."
                )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
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
        # Check for custom GPU mapping
        gpu_mapping = self.config.get("training", {}).get("model_gpu_mapping", {})

        for i, model_config in enumerate(self.model_configs):
            model_id = f"M{i+1}"
            model_name = model_config["name"]

            model, tokenizer = self.load_model(model_name, model_id)

            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer

            # Create reference model (frozen copy) for KL divergence
            use_mistral3 = is_mistral3_model(model_name) and MISTRAL3_AVAILABLE

            # Determine device_map for reference model
            if use_mistral3:
                ref_device_map = "auto"
            else:
                gpu_ids = gpu_mapping.get(model_id, None)
                if gpu_ids:
                    ref_device_map = build_device_map_for_gpus(model_name, gpu_ids)
                else:
                    ref_device_map = "auto"

            # For Mistral3, use 4-bit quantization for reference model to save memory
            if use_mistral3:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info(f"Loading reference model {model_id} with 4-bit quantization to save memory")
                ref_model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    device_map=ref_device_map,
                )
            else:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
                    trust_remote_code=True,
                    device_map=ref_device_map,
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
        use_multi_strategy: bool = None,
    ) -> List[List[str]]:
        """
        Generate reasoning traces for questions.

        Args:
            model_id: Which model to use
            questions: List of questions
            num_traces: Number of traces per question
            context: Optional context to prepend (for rescue/contexted generation)
            use_multi_strategy: Override multi_strategy setting (None = use config)

        Returns:
            List of trace lists (one list per question)
        """
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        model.eval()

        # Determine if using multi-strategy
        multi_strategy = use_multi_strategy if use_multi_strategy is not None else self.multi_strategy

        all_traces = []

        for question in questions:
            # Prepare prompt based on mode
            if multi_strategy:
                if context:
                    # Multi-strategy with context (rescue)
                    prompt = format_multi_strategy_contexted_prompt(question, context)
                else:
                    # Multi-strategy cold generation
                    prompt = format_multi_strategy_prompt(question)
            else:
                # Standard prompt
                if context:
                    prompt = f"{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"
                else:
                    prompt = format_prompt(question)

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

    def get_prompt_for_question(
        self,
        question: str,
        context: str = None,
    ) -> str:
        """
        Get the appropriate prompt for a question.

        Args:
            question: The question to solve
            context: Optional context (for rescue generation)

        Returns:
            Formatted prompt string
        """
        if self.multi_strategy:
            if context:
                return format_multi_strategy_contexted_prompt(question, context)
            else:
                return format_multi_strategy_prompt(question)
        else:
            if context:
                return f"{context}\n\nQuestion: {question}\n\nLet's solve this step by step:"
            else:
                return format_prompt(question)

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
        """Save training checkpoint with verification."""
        # Use absolute path to avoid any working directory issues
        checkpoint_dir = self.output_dir.resolve() / f"checkpoint-{tag}"

        # Ensure parent directory exists first
        self.output_dir.resolve().mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir} (absolute: {checkpoint_dir.resolve()})")

        saved_files = []

        # Save models
        for model_id, model in self.models.items():
            model_dir = checkpoint_dir / model_id
            model.save_pretrained(model_dir)
            self.tokenizers[model_id].save_pretrained(model_dir)
            saved_files.append(model_dir)

        # Save training state
        state_dict = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_accuracy": self.state.best_accuracy,
            "best_rescue_rate": self.state.best_rescue_rate,
            "metrics_history": self.state.metrics_history,
        }

        state_file = checkpoint_dir / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
        saved_files.append(state_file)

        # Save optimizer states
        for model_id, optimizer in self.optimizers.items():
            opt_file = checkpoint_dir / f"optimizer_{model_id}.pt"
            torch.save(optimizer.state_dict(), opt_file)
            saved_files.append(opt_file)

        # Verify checkpoint was saved
        if checkpoint_dir.exists():
            file_count = sum(1 for _ in checkpoint_dir.rglob("*") if _.is_file())
            logger.info(f"Checkpoint verified: {checkpoint_dir} ({file_count} files)")
        else:
            logger.error(f"CHECKPOINT VERIFICATION FAILED: {checkpoint_dir} does not exist after save!")

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
