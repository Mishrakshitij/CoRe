"""Auditable, single-process CoRe training with Hugging Face and PEFT.

Each model lives on one explicitly selected device. Rollouts are generated before
any optimizer step; old/reference token log probabilities are frozen for the
whole team. This is a reference implementation, not the legacy distributed
runner. Torch, Transformers and PEFT are imported only when training starts.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .parsing import answers_equal, extract_answer
from .protocol import build_hint, build_prompt, select_teacher
from .rewards import (
    FrozenSentenceEncoder,
    RewardConfig,
    Sample,
    overlap_partial_credit,
    score_question_groups,
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_name: str
    device: str = "cuda:0"
    revision: str | None = None


@dataclass
class TrainConfig:
    models: list[ModelConfig]
    seed: int = 42
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    cold_samples: int = 2
    context_samples: int = 1
    hint_probability: float = 0.75
    hint_tokens: int = 1536
    max_prompt_tokens: int = 4096
    max_new_tokens: int = 3072
    temperature: float = 0.7
    top_p: float = 0.9
    round_b_weight: float = 0.8
    algorithm: str = "grpo"
    likelihood_mode: str = "raw_policy_surrogate"
    clip_epsilon: float = 0.2
    clip_epsilon_high: float = 0.2
    kl_beta: float = 0.04
    policy_updates_per_batch: int = 1
    teacher_score: str = "reward"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    partial_credit: str = "token_f1"
    reward: dict[str, Any] = field(default_factory=dict)
    reward_overrides_by_epoch: list[dict[str, Any]] = field(default_factory=list)
    output_dir: str = "outputs/core-paper"
    save_every_steps: int = 100
    log_rollouts: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        raw = dict(values)
        raw["models"] = [ModelConfig(**item) for item in raw["models"]]
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.models) < 2:
            raise ValueError("CoRe requires at least two independently trainable models")
        ids = [model.model_id for model in self.models]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
            raise ValueError("model_id values must be unique safe directory names")
        if self.algorithm != "grpo":
            raise ValueError("This reference trainer implements GRPO only; GSPO/SAPO are not implemented")
        if self.likelihood_mode != "raw_policy_surrogate":
            raise ValueError("likelihood_mode must be raw_policy_surrogate; scores use unwarped model probabilities")
        for key in ("epochs", "batch_size", "cold_samples", "context_samples", "hint_tokens",
                    "max_prompt_tokens", "max_new_tokens", "policy_updates_per_batch",
                    "lora_rank", "lora_alpha", "save_every_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "max_grad_norm", "temperature"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("weight_decay", "round_b_weight", "kl_beta"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        for key in ("hint_probability", "warmup_ratio", "lora_dropout"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < self.clip_epsilon < 1 or self.clip_epsilon_high <= 0:
            raise ValueError("Clipping requires 0 < clip_epsilon < 1 and clip_epsilon_high > 0")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if self.dtype == "float16":
            raise ValueError("float16 requires loss scaling, which this trainer does not implement; use bfloat16 or float32")
        if self.teacher_score not in {"reward", "exploit_reward"}:
            raise ValueError("teacher_score must be reward or exploit_reward")
        if self.partial_credit not in {"token_f1", "none"}:
            raise ValueError("partial_credit must be token_f1 or none")
        if self.lora_dropout != 0:
            raise ValueError("Use lora_dropout=0 so rollout and update likelihoods describe the same deterministic policy")
        for epoch in range(self.epochs):
            reward = self.rewards_for_epoch(epoch)
            if self.partial_credit == "none" and reward.alpha:
                raise ValueError("partial_credit='none' requires reward alpha=0 for every epoch")

    def rewards_for_epoch(self, epoch: int) -> RewardConfig:
        values = dict(self.reward)
        if self.reward_overrides_by_epoch:
            values.update(self.reward_overrides_by_epoch[min(epoch, len(self.reward_overrides_by_epoch) - 1)])
        return RewardConfig(**values)


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str


def load_jsonl(path: str | Path) -> list[Example]:
    """Read an already selected training split; never silently create a split."""
    examples: list[Example] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not all(key in item for key in ("id", "question", "answer")):
                raise ValueError(f"Line {line_number}: required fields are id, question, answer")
            question, answer = item["question"], item["answer"]
            raw_identity = item["id"]
            if not isinstance(raw_identity, (str, int)) or isinstance(raw_identity, bool):
                raise ValueError(f"Line {line_number}: id must be nonempty text or an integer")
            identity = str(raw_identity)
            if not identity.strip() or identity != identity.strip() or identity in seen:
                raise ValueError(f"Line {line_number}: empty or duplicate example id {identity!r}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number}: question must be nonempty text")
            if not isinstance(answer, (str, int, float)) or isinstance(answer, bool) or not str(answer).strip():
                raise ValueError(f"Line {line_number}: answer must be nonempty text or a number")
            if isinstance(answer, float) and not math.isfinite(answer):
                raise ValueError(f"Line {line_number}: answer must be finite")
            seen.add(identity)
            examples.append(Example(identity, question, str(answer)))
    if not examples:
        raise ValueError("Training file contains no examples")
    return examples


@dataclass
class Generation:
    text: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]


@dataclass
class Rollout:
    sample: Sample
    prompt: str
    generation: Generation
    old_log_probs: Any = None
    reference_log_probs: Any = None


class PolicyBackend(Protocol):
    model_id: str
    tokenizer: Any

    def generate(self, prompt: str) -> Generation: ...
    def snapshot(self, rollout: Rollout) -> None: ...
    def update(self, rollouts: list[Rollout]) -> dict[str, float]: ...
    def save(self, path: Path) -> None: ...


def load_model_and_tokenizer(specification: ModelConfig, dtype: Any,
                             tokenizer_source: str | None = None) -> tuple[Any, Any]:
    """Load a supported paper architecture, including text-only use of Ministral 3.

    Returns (model, tokenizer). No remote Python code is executed. The model is
    placed on the specification's single device; distributed placement is not
    part of this reference implementation.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    common = {"revision": specification.revision, "trust_remote_code": False}
    configuration = AutoConfig.from_pretrained(specification.model_name, **common)
    source = tokenizer_source or specification.model_name
    tokenizer_options = {"trust_remote_code": False}
    if tokenizer_source is None:
        tokenizer_options["revision"] = specification.revision
    if configuration.model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        model_class = Mistral3ForConditionalGeneration
        tokenizer = MistralCommonBackend.from_pretrained(source, **tokenizer_options)
    else:
        model_class = AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_options)
    model = model_class.from_pretrained(specification.model_name, torch_dtype=dtype, **common).to(specification.device)
    return model, tokenizer


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Render chat templates when available, retaining exact token boundaries."""
    uses_native_mistral_template = tokenizer.__class__.__name__ == "MistralCommonBackend"
    if getattr(tokenizer, "chat_template", None) or uses_native_mistral_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
    return tokenizer.encode(prompt, add_special_tokens=True)


def grpo_loss(current: Any, old: Any, reference: Any, advantage: float,
              clip_epsilon: float, clip_epsilon_high: float, kl_beta: float) -> tuple[Any, Any]:
    """Equation 21 on the actual completion tokens of one trajectory.

    PPO's minimum applies for *both* advantage signs. Old/reference tensors are
    detached unconditionally. The reference penalty uses the usual nonnegative
    sampled k3 estimator exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1.
    """
    import torch

    if current.ndim != 1 or current.numel() == 0 or old.shape != current.shape or reference.shape != current.shape:
        raise ValueError("Log probabilities must be equal-length, nonempty completion-token vectors")
    old = old.detach().to(current.device)
    reference = reference.detach().to(current.device)
    ratio = torch.exp(current - old)
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon_high)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    delta = reference - current
    kl = torch.expm1(delta) - delta
    return -surrogate.mean() + kl_beta * kl.mean(), kl.mean()


class HFCausalPolicy:
    """One new LoRA policy; disabling adapters gives the frozen initial reference."""

    def __init__(self, specification: ModelConfig, config: TrainConfig, total_updates: int):
        import torch
        from peft import LoraConfig, get_peft_model

        self.model_id = specification.model_id
        self.specification, self.config = specification, config
        self.device = torch.device(specification.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"{self.model_id} requires {self.device}; set device='cpu' and dtype='float32' for a tiny-model smoke run")
        base, self.tokenizer = load_model_and_tokenizer(specification, getattr(torch, config.dtype))
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(f"{self.model_id}: tokenizer must provide an EOS or PAD token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        adapter_config = LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules, task_type="CAUSAL_LM", bias="none",
        )
        self.model = get_peft_model(base, adapter_config)
        # Policy likelihoods must not change merely because train/eval mode did.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            # HF attention implementations may use functional dropout from a
            # numeric module attribute rather than an nn.Dropout child.
            for name in ("attention_dropout", "hidden_dropout", "resid_pdrop", "attn_pdrop"):
                if isinstance(getattr(module, name, None), (int, float)):
                    setattr(module, name, 0.0)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable adapter parameters")
        self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
        warmup = int(total_updates * config.warmup_ratio)

        def learning_rate_scale(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total_updates - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, learning_rate_scale)

    def _encode_prompt(self, prompt: str) -> list[int]:
        ids = encode_prompt(self.tokenizer, prompt)
        if not ids:
            raise ValueError("Tokenizer produced an empty prompt")
        if len(ids) > self.config.max_prompt_tokens:
            raise ValueError(f"Prompt has {len(ids)} tokens, above max_prompt_tokens={self.config.max_prompt_tokens}; shorten the input or raise the explicit cap")
        model_config = getattr(self.model.config, "text_config", self.model.config)
        context_window = getattr(model_config, "max_position_embeddings", None)
        if context_window and len(ids) + self.config.max_new_tokens > context_window:
            raise ValueError("Prompt plus generation budget exceeds the model context window")
        return ids

    def generate(self, prompt: str) -> Generation:
        import torch

        self.model.eval()
        ids = self._encode_prompt(prompt)
        inputs = torch.tensor([ids], device=self.device, dtype=torch.long)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=True, temperature=self.config.temperature, top_p=self.config.top_p,
                top_k=0, max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=1, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0, len(ids):].tolist()
        if not generated:
            raise RuntimeError("Model generated no completion tokens")
        # Single-sequence decoding avoids padded completions; EOS remains in the loss.
        return Generation(self.tokenizer.decode(generated, skip_special_tokens=True), tuple(ids), tuple(generated))

    def _log_probs(self, rollout: Rollout) -> Any:
        # These are raw policy probabilities, not probabilities after the
        # temperature/top-p sampling warpers. That conventional rollout
        # surrogate is explicit in config.likelihood_mode; it is not claimed
        # to be an exact importance correction for the decoding distribution.
        import torch

        prompt_ids = rollout.generation.prompt_ids
        completion_ids = rollout.generation.completion_ids
        tokens = torch.tensor([prompt_ids + completion_ids], device=self.device, dtype=torch.long)
        output = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
        # The last prompt position predicts the first completion token. Never
        # re-tokenize decoded text or substitute a cold prompt for a hinted one.
        logits = output.logits[0, len(prompt_ids) - 1:-1].float()
        targets = tokens[0, len(prompt_ids):]
        return logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)

    def snapshot(self, rollout: Rollout) -> None:
        import torch

        self.model.eval()
        with torch.no_grad():
            rollout.old_log_probs = self._log_probs(rollout).detach().cpu()
            with self.model.disable_adapter():
                rollout.reference_log_probs = self._log_probs(rollout).detach().cpu()

    def update(self, rollouts: list[Rollout]) -> dict[str, float]:
        import torch

        if not rollouts or any(r.old_log_probs is None or r.reference_log_probs is None for r in rollouts):
            raise ValueError("Every rollout must have a frozen old/reference snapshot before update")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = kl_total = 0.0
        for rollout in rollouts:
            current = self._log_probs(rollout)
            loss, kl = grpo_loss(
                current, rollout.old_log_probs, rollout.reference_log_probs,
                rollout.sample.advantage, self.config.clip_epsilon,
                self.config.clip_epsilon_high, self.config.kl_beta,
            )
            weight = self.config.round_b_weight if rollout.sample.round_id == "B" else 1.0
            contribution = weight * loss / len(rollouts)
            if not torch.isfinite(contribution):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Nonfinite policy loss for {self.model_id}")
            contribution.backward()
            loss_total += contribution.detach().item()
            kl_total += weight * kl.detach().item() / len(rollouts)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss_total, "kl": kl_total, "gradient_norm": float(norm),
                "learning_rate": self.optimizer.param_groups[0]["lr"]}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class CoReTrainer:
    def __init__(self, config: TrainConfig, backends: list[PolicyBackend], encoder: Any):
        config.validate()
        if [backend.model_id for backend in backends] != [model.model_id for model in config.models]:
            raise ValueError("Backend model identities must match configured model order")
        self.config, self.backends, self.encoder = config, backends, encoder
        self.random = random.Random(config.seed)
        self.partial_credit_fn = overlap_partial_credit if config.partial_credit == "token_f1" else None
        self._diagnostics: list[Any] = []

    def _sample(self, backend: PolicyBackend, example: Example, round_id: str,
                hint: str | None = None, cold_succeeded: bool = False) -> Rollout:
        prompt = build_prompt(example.question, hint=hint)
        generation = backend.generate(prompt)
        answer = extract_answer(generation.text)
        sample = Sample(
            question_id=example.id, model_id=backend.model_id, round_id=round_id,
            text=generation.text, answer=answer or "", correct=answers_equal(answer, example.answer),
            hint_provided=bool(hint), cold_succeeded=cold_succeeded, gold=example.answer,
        )
        return Rollout(sample, prompt, generation)

    def collect_question(self, example: Example, reward_config: RewardConfig) -> list[Rollout]:
        cold = [self._sample(backend, example, "A") for backend in self.backends
                for _ in range(self.config.cold_samples)]
        score_question_groups([r.sample for r in cold], reward_config, encoder=self.encoder,
                              partial_credit_fn=self.partial_credit_fn)
        teacher = select_teacher([r.sample for r in cold], score=self.config.teacher_score)
        successes = {backend.model_id: any(r.sample.correct for r in cold if r.sample.model_id == backend.model_id)
                     for backend in self.backends}
        contexted: list[Rollout] = []
        for backend in self.backends:
            candidate_hint = (build_hint(teacher.text, example.answer, token_budget=self.config.hint_tokens,
                                         tokenizer=backend.tokenizer) if teacher is not None else None)
            for _ in range(self.config.context_samples):
                hint = candidate_hint if self.random.random() < self.config.hint_probability else None
                contexted.append(self._sample(backend, example, "B", hint, successes[backend.model_id]))
        rollouts = cold + contexted
        self._diagnostics.extend(score_question_groups(
            [r.sample for r in rollouts], reward_config, encoder=self.encoder,
            partial_credit_fn=self.partial_credit_fn,
        ))
        return rollouts

    def train_batch(self, examples: list[Example], reward_config: RewardConfig) -> tuple[list[Rollout], dict[str, Any]]:
        if not examples:
            raise ValueError("Cannot train an empty batch")
        self._diagnostics = []
        rollouts = [rollout for example in examples for rollout in self.collect_question(example, reward_config)]
        by_model = {backend.model_id: [r for r in rollouts if r.sample.model_id == backend.model_id]
                    for backend in self.backends}
        # This barrier is essential: no policy may change until all models'
        # behavior/reference likelihoods for both rounds have been detached.
        for backend in self.backends:
            for rollout in by_model[backend.model_id]:
                backend.snapshot(rollout)
        metrics: dict[str, Any] = {}
        for _ in range(self.config.policy_updates_per_batch):
            for backend in self.backends:
                metrics[backend.model_id] = backend.update(by_model[backend.model_id])
        metrics["mean_reward"] = sum(r.sample.reward for r in rollouts) / len(rollouts)
        metrics["cold_accuracy"] = sum(r.sample.correct for r in rollouts if r.sample.round_id == "A") / sum(r.sample.round_id == "A" for r in rollouts)
        eligible = [r for r in rollouts if r.sample.round_id == "B" and r.sample.hint_provided and not r.sample.cold_succeeded]
        metrics["rescue_eligible"] = len(eligible)
        metrics["rescued"] = sum(r.sample.correct for r in eligible)
        metrics["exploration_all_zero_groups"] = sum(d.exploration_all_zero for d in self._diagnostics)
        metrics["question_groups"] = len(self._diagnostics)
        return rollouts, metrics

    def fit(self, examples: list[Example]) -> None:
        destination = Path(self.config.output_dir)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Output directory is nonempty: {destination}; choose a new run directory")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8")
        packages = {}
        for package in ("torch", "transformers", "peft", "numpy", "sentence-transformers", "mistral-common"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "unavailable"
        (destination / "environment.json").write_text(json.dumps(packages, indent=2) + "\n", encoding="utf-8")
        canonical = json.dumps([asdict(example) for example in examples], sort_keys=True, ensure_ascii=False)
        manifest = {"num_examples": len(examples), "ids": [x.id for x in examples],
                    "canonical_data_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                    "note": "Caller-supplied split; this run does not establish paper split identity."}
        (destination / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        step = 0
        for epoch in range(self.config.epochs):
            order = list(examples)
            self.random.shuffle(order)
            for offset in range(0, len(order), self.config.batch_size):
                batch = order[offset:offset + self.config.batch_size]
                rollouts, metrics = self.train_batch(batch, self.config.rewards_for_epoch(epoch))
                step += 1
                metrics.update({"step": step, "epoch": epoch + 1, "example_ids": [x.id for x in batch]})
                with (destination / "metrics.jsonl").open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                if self.config.log_rollouts:
                    with (destination / "rollouts.jsonl").open("a", encoding="utf-8") as sink:
                        for rollout in rollouts:
                            sink.write(json.dumps({"step": step, "sample": asdict(rollout.sample),
                                                   "prompt": rollout.prompt, "generation": asdict(rollout.generation)},
                                                  ensure_ascii=False) + "\n")
                if step % self.config.save_every_steps == 0:
                    self.save(destination / f"checkpoint-{step}")
        self.save(destination / "final")

    def save(self, destination: Path) -> None:
        for backend in self.backends:
            backend.save(destination / backend.model_id)
        # Adapter exports support inference and fresh optimizer restarts. Exact
        # optimizer/RNG resumption is deliberately not advertised or implemented.


def run_training(config: TrainConfig, examples: list[Example]) -> None:
    import torch
    from transformers import set_seed

    config.validate()
    destination = Path(config.output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {destination}")
    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    updates = math.ceil(len(examples) / config.batch_size) * config.epochs * config.policy_updates_per_batch
    backends = [HFCausalPolicy(model, config, updates) for model in config.models]
    encoder = FrozenSentenceEncoder(model_name=config.embedding_model, device=config.embedding_device)
    CoReTrainer(config, backends, encoder).fit(examples)
