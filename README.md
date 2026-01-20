# Collaborative Reasoning with GRPO/GSPO/SAPO

A framework for training multiple language models to collaboratively solve reasoning problems using policy optimization.

## Results

### GSM8K Test Set (1319 samples)

| Algorithm | Checkpoint | M1 (Qwen2.5-3B) | M2 (Qwen3-4B) | Combined | Collab Gain | Rescue Rate |
|-----------|------------|-----------------|---------------|----------|-------------|-------------|
| **GRPO** | step_500 | 92.12% | 92.12% | 98.94% | +6.82% | 100% |
| **GRPO** | step_1000 | 92.95% | 92.80% | 99.62% | +6.67% | 100% |
| **GRPO** | step_1500 | 92.95% | 93.33% | 99.92% | +6.60% | 100% |
| **GSPO** | step_1000 | 93.48% | 93.56% | 99.47% | +5.91% | 100% |

**Key Metrics:**
- **Combined Accuracy**: When either M1 or M2 is correct
- **Collaboration Gain**: Combined accuracy minus best single model accuracy
- **Rescue Rate**: When models disagree, how often at least one is correct

## Overview

This project implements **Collaborative Reasoning** where multiple models (M1, M2, ..., MN) learn to reason together through:

1. **Explore-Exploit Rewards**: Models are rewarded for both correctness (exploit) and diverse reasoning strategies (explore)
2. **Cross-Model Teaching**: Correct reasoning traces from one model help guide other models
3. **Two-Epoch Training**: Exploration in epoch 1, exploitation in epoch 2 with distillation

### Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **GRPO** | Group Relative Policy Optimization (token-level) | Baseline |
| **GSPO** | Group Sequence Policy Optimization (sequence-level) | MoE models, stability |
| **SAPO** | Soft Adaptive Policy Optimization (soft gating) | Sample efficiency |
| **Hybrid** | GSPO + SAPO combination | Collaborative training |

### Supported Models

| Model Key | Model Name | Type | Size | Notes |
|-----------|------------|------|------|-------|
| `qwen2_5_3b` | Qwen/Qwen2.5-3B-Instruct | qwen | 3B | Standard |
| `qwen3_4b` | Qwen/Qwen3-4B-Instruct-2507 | qwen | 4B | Standard |
| `llama3_2_3b` | meta-llama/Llama-3.2-3B-Instruct | llama | 3B | Standard |
| `ministral_3b` | ministral/Ministral-3b-instruct | mistral | 3B | Standard |
| `ministral_3b_reasoning` | mistralai/Ministral-3-3B-Reasoning-2512 | mistral3 | 3B | **Requires transformers>=5.0.0** |
| `ministral_8b_reasoning` | mistralai/Ministral-3-8B-Reasoning-2512 | mistral3 | 8B | **Requires transformers>=5.0.0** |
| `ministral_14b_reasoning` | mistralai/Ministral-3-14B-Reasoning-2512 | mistral3 | 14B | **Requires transformers>=5.0.0, QLoRA** |
| `phi4_reasoning` | microsoft/Phi-4-reasoning | phi4 | 14B | QLoRA recommended |
| `phi4_reasoning_plus` | microsoft/Phi-4-reasoning-plus | phi4 | 14B | QLoRA recommended |

**Mistral-3 Reasoning Models:**
The Ministral reasoning models use `Mistral3ForConditionalGeneration` which requires transformers 5.0.0+. Set up a separate conda environment:

```bash
# Create environment for Mistral-3 models
conda create -n mistral_env python=3.11 -y
conda activate mistral_env

pip install transformers==5.0.0rc0
pip install mistral-common>=1.8.6
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install accelerate peft datasets wandb tqdm PyYAML

# Train with Mistral reasoning models (basic - uses chat template)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_fast.py \
  --config configs/grpo_gpqa_mistral.yaml \
  --prompt-template mistral-chat \
  --no-compile

# Train with Mistral reasoning models (full features - with [THINK] reward)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_fast.py \
  --config configs/grpo_gpqa_mistral.yaml \
  --prompt-template mistral-chat \
  --think-reward \
  --no-compile
```

**Prompt Template Options:**
- `--prompt-template standard`: Default raw prompts (works for Qwen, Llama, etc.)
- `--prompt-template mistral-chat`: Uses `apply_chat_template()` for Mistral-3 models to trigger `[THINK]...[/THINK]` reasoning format
- `--prompt-template phi-chat`: Uses ChatML template for Phi-4 models to trigger `<think>...</think>` reasoning format
- `--prompt-template auto`: Auto-detect template based on model name (recommended for mixed model training)
- `--think-reward`: Adds diversity reward for thinking blocks (works with both `[THINK]` and `<think>` formats)

**Phi-4 Reasoning Models:**
Microsoft's Phi-4 reasoning models use `<think>...</think>` tags for reasoning:

```bash
# Train with Phi-4 reasoning model
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_fast.py \
  --config configs/grpo_gpqa_phi4_ministral14b_qlora.yaml \
  --prompt-template phi-chat \
  --think-reward \
  --no-compile
```

**QLoRA Training for Large Models (14B+):**
For 14B+ models, QLoRA is essential to fit on 40GB GPUs:

```bash
# Train Phi-4 + Ministral-14B with QLoRA (requires 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_fast.py \
  --config configs/grpo_gpqa_phi4_ministral14b_qlora.yaml \
  --prompt-template auto \
  --think-reward \
  --gen-batch-size 1 \
  --no-compile
```

QLoRA config options in YAML:
```yaml
training:
  use_qlora: true           # Enable QLoRA
  qlora_bits: 4             # 4-bit (default) or 8-bit quantization
  max_memory_per_gpu: "38GB"  # Memory limit per GPU
  lora_r: 32                # Higher rank for larger models
```

**Per-Model Generation Config:**
Each model can have its own generation settings:
```yaml
models:
  available:
    phi4_reasoning:
      name: "microsoft/Phi-4-reasoning-plus"
      generation:           # Model-specific overrides
        temperature: 0.8
        top_k: 50
        top_p: 0.95
    ministral_14b:
      name: "mistralai/Ministral-3-14B-Reasoning-2512"
      generation:
        temperature: 0.7
        top_p: 0.9
```

**Auto-Detection of Prompt Format:**
When using `--prompt-template auto`, the system automatically detects the appropriate template based on model name and dataset:

| Dataset | System Message | Answer Format |
|---------|----------------|---------------|
| **AIME** | Olympiad mathematician (algebraic, combinatorial, geometric, number theory) | Integer (000-999) |
| **GSM8K** | Grade-school math solver | Numerical |
| **MATH** | Competition math solver | Numerical |
| **GPQA** | Expert scientist (scientific principles, process of elimination) | MCQ (A/B/C/D) |
| **MedMCQA** | Medical professional (clinical reasoning) | MCQ (A/B/C/D) |

No need to specify answer format - it's determined by the dataset config!

### Supported Datasets

| Dataset | Domain | Answer Format | Notes |
|---------|--------|---------------|-------|
| **GSM8K** | Grade-school math | Numerical | 7.5K train, 1.3K test |
| **MATH** | Competition math | LaTeX boxed | Uses qwedsacf/competition_math |
| **AIME** | AMC/AIME competition | Integer (000-999) | AI-MO/aimo-validation-aime |
| **GPQA** | Graduate-level science | MCQ (A/B/C/D) | Gated - requires HF auth |
| **MedMCQA** | Medical entrance exam | MCQ (A/B/C/D) | 182K+ samples |

**Using Different Datasets:**
```bash
# GSM8K (default)
python scripts/train_fast.py --config configs/grpo_fast_1000.yaml --dataset gsm8k

# MATH (competition math)
python scripts/train_fast.py --config configs/grpo_full_math.yaml --dataset math_qwedsacf

# AIME (competition)
python scripts/train_fast.py --config configs/base_config.yaml --dataset aime

# MedMCQA (medical)
python scripts/train_fast.py --config configs/base_config.yaml --dataset medmcqa
```

**Note on GPQA:** GPQA is a gated dataset. To use it:
1. Run `huggingface-cli login`
2. Request access at https://huggingface.co/datasets/Idavidrein/gpqa

## Installation

```bash
# Clone repository
git clone <repo-url>
cd colab_reason_v1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Fast Training (Recommended)

The fast trainer uses batched generation and optimized inference for 3-5x speedup:

```bash
# Fast GRPO training on GSM8K
python scripts/train_fast.py \
    --config configs/grpo_fast_1000.yaml \
    --algorithm grpo \
    --num-samples 7472 \
    --gen-batch-size 4 \
    --use-wandb

# Disable torch.compile for debugging
python scripts/train_fast.py \
    --config configs/grpo_fast_1000.yaml \
    --no-compile
```

**Fast Training Options:**
- `--gen-batch-size`: Batch size for generation (default: 4, higher = faster but more memory)
- `--no-compile`: Disable torch.compile for debugging
- `--use-wandb`: Enable Weights & Biases logging

### Standard Training

```bash
# Train Qwen2.5-3B + Qwen3-4B with GRPO
python scripts/train.py \
    --config configs/base_config.yaml \
    --algorithm grpo \
    --dataset gsm8k \
    --use-wandb
```

### Using Experiment Configs

```bash
# Run specific experiment
python scripts/train.py \
    --config configs/base_config.yaml \
    --experiment configs/experiments/pairwise_qwen_llama.yaml
```

### Running All Experiments

```bash
# Dry run (see commands without executing)
python scripts/run_all_experiments.py --experiments all --dry-run

# Run all experiments
python scripts/run_all_experiments.py --experiments pairwise
```

## Project Structure

```
colab_reason/
├── configs/
│   ├── base_config.yaml           # Base configuration
│   ├── grpo_fast_1000.yaml        # Fast training config
│   └── experiments/               # Experiment-specific configs
├── src/
│   ├── losses/                    # GRPO, GSPO, SAPO implementations
│   │   ├── grpo_loss.py
│   │   ├── gspo_loss.py
│   │   ├── sapo_loss.py
│   │   └── hybrid_loss.py
│   ├── rewards/                   # Reward functions
│   │   ├── exploit_reward.py      # Correctness-based
│   │   ├── explore_reward.py      # DPP-lite diversity
│   │   ├── cross_reward.py        # Cross-model complementarity
│   │   ├── think_reward.py        # [THINK] diversity (Mistral)
│   │   └── combined_reward.py     # Combined reward function
│   ├── trainers/                  # Training logic
│   │   ├── base_trainer.py
│   │   ├── collab_trainer.py      # Main collaborative trainer
│   │   ├── fast_trainer.py        # Fast trainer with batched generation
│   │   ├── micro_rounds.py        # Micro-round A/B logic
│   │   └── buddy_buffer.py        # Cross-teaching buffer
│   └── data/                      # Data loading
│       ├── dataset.py
│       └── preprocessing.py
├── scripts/
│   ├── train.py                   # Standard training script
│   ├── train_fast.py              # Fast training script (3-5x speedup)
│   └── run_all_experiments.py     # Batch experiment runner
├── evaluation/
│   ├── evaluate.py                # Standard evaluation script
│   ├── evaluate_vllm_single.py    # vLLM-based fast evaluation (single model)
│   └── combine_vllm_results.py    # Combine M1/M2 results for collaboration metrics
└── outputs/                       # Checkpoints and logs
```

## Training Flow

### Micro-Round Training (Per Question)

```
┌─────────────────────────────────────────────────────────────┐
│ Micro-Round A (Cold Generation)                              │
├─────────────────────────────────────────────────────────────┤
│ • M1 generates K traces for question Q                       │
│ • M2 generates K traces for question Q                       │
│ • Compute explore + exploit rewards                          │
│ • Identify best correct trace (if any)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Context Compression                                          │
├─────────────────────────────────────────────────────────────┤
│ • Compress best trace to teacher context (≤120 tokens)       │
│ • Extract: approach signature + key insights                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Micro-Round B (Contexted Generation)                         │
├─────────────────────────────────────────────────────────────┤
│ • Hint dropout (p=0.5): some traces see context              │
│ • M1 generates K' traces with/without context                │
│ • M2 generates K' traces with/without context                │
│ • Compute rewards + rescue bonus if applicable               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GRPO/GSPO/SAPO Update                                        │
├─────────────────────────────────────────────────────────────┤
│ • Combine A+B traces with rewards                            │
│ • Compute advantages (group-normalized)                      │
│ • Update M1, M2 with policy gradient                         │
│ • Store successful rescues in buddy buffer                   │
└─────────────────────────────────────────────────────────────┘
```

## Reward Components

### Exploitation Reward (R_exploit)
```
R_exploit(τ) = 1.0 if correct else α × partial_score
```

### Exploration Reward (R_explore) - DPP-Lite
```
R_explore(τ) = max(0, min_{τ'∈S} d(τ,τ') - δ)
```
Where S is a greedily constructed diverse set.

### Cross-Model Reward (R_cross)
```
R_cross(τ) = η × min_{τ'∈partner} d(τ,τ')
Only if R_exploit(τ) ≥ quality_threshold
```

### Think Reward (R_think) - Mistral Only
```
R_think(τ) = w_think × diversity([THINK] content)
Only active with --think-reward flag for Mistral reasoning models
```

### Combined Reward
```
R(τ) = w_explt × R_exploit + w_exp × R_explore + w_cross × R_cross + R_think + rescue_bonus
```

## Evaluation

### vLLM Evaluation (Recommended - Fast)

The vLLM-based evaluator provides 10-20x faster inference using optimized batched generation:

```bash
# Step 1: Evaluate M1 (Qwen2.5-3B)
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_vllm_single.py \
    --checkpoint outputs/experiment_name/checkpoint-step_1000 \
    --model M1 \
    --dataset gsm8k \
    --split test \
    --output results_M1.json \
    --gpu-memory 0.45

# Step 2: Evaluate M2 (Qwen3-4B)
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_vllm_single.py \
    --checkpoint outputs/experiment_name/checkpoint-step_1000 \
    --model M2 \
    --dataset gsm8k \
    --split test \
    --output results_M2.json \
    --gpu-memory 0.45

# Step 3: Combine results for collaboration metrics
python evaluation/combine_vllm_results.py \
    --m1 results_M1.json \
    --m2 results_M2.json \
    --output results_combined.json
```

**vLLM Options:**
- `--model`: Which model to evaluate (`M1` for Qwen2.5-3B, `M2` for Qwen3-4B)
- `--gpu-memory`: Fraction of GPU memory to use (default: 0.9, use 0.45 to run 2 models on same GPU)
- `--batch-size`: Batch size for vLLM inference (default: 128)

### Standard Evaluation

```bash
# Evaluate both models together (slower but simpler)
python evaluation/evaluate.py \
    --checkpoint outputs/experiment_name/checkpoint-final \
    --dataset gsm8k \
    --split test \
    --output results.json
```

### Metrics
- **Accuracy**: Per-model and combined (any model correct)
- **Rescue Rate**: P[B correct | A failed] - when models disagree, how often at least one is correct
- **Collaboration Gain**: Combined accuracy - Best single model accuracy
- **Diversity**: Average pairwise trace distance, unique strategies

## Model Combinations

### Pairwise (6 combinations)
| Pair | Models | Family |
|------|--------|--------|
| 1 | Qwen2.5-3B + Qwen3-4B | Same |
| 2 | Qwen2.5-3B + Llama-3.2-3B | Cross |
| 3 | Qwen2.5-3B + Ministral-3B | Cross |
| 4 | Qwen3-4B + Llama-3.2-3B | Cross |
| 5 | Qwen3-4B + Ministral-3B | Cross |
| 6 | Llama-3.2-3B + Ministral-3B | Cross |

### Multi-Model
- **Trios**: 3 models collaborating
- **Quad**: All 4 models collaborating

## Configuration

Key configuration options in `configs/base_config.yaml`:

```yaml
policy_optimization:
  algorithm: "gspo_sapo_hybrid"  # grpo, gspo, sapo, gspo_sapo_hybrid
  K: 4           # Traces per cold round
  K_prime: 2     # Traces per contexted round
  beta: 0.04     # KL coefficient
  epsilon: 3e-4  # GSPO clipping
  tau_pos: 1.0   # SAPO temperature (positive)
  tau_neg: 1.05  # SAPO temperature (negative)

rewards:
  w_exploit_e1: 1.0   # Epoch 1 exploit weight
  w_explore_e1: 0.2   # Epoch 1 explore weight
  w_exploit_e2: 1.0   # Epoch 2 exploit weight
  w_explore_e2: 0.05  # Epoch 2 explore weight (annealed)
  delta: 0.15         # DPP-lite margin
  r_teach: 0.15       # Rescue bonus

collaboration:
  p_hint: 0.5              # Hint dropout probability
  max_context_tokens: 120  # Teacher context length
  enable_distillation: false  # Optional: disable epoch-2 distillation (GRPO already learns from hints)
```

## References

- **GRPO**: DeepSeek-R1 (2024)
- **GSPO**: [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) - Qwen Team
- **SAPO**: [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347) - Qwen Team

## License

MIT License
