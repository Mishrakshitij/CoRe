# Collaborative Reasoning with GRPO/GSPO/SAPO

A framework for training multiple language models to collaboratively solve reasoning problems using policy optimization.

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

### Training a Pairwise Collaboration

```bash
# Train Qwen2.5-3B + Llama-3.2-3B with GSPO-SAPO hybrid
python scripts/train.py \
    --config configs/base_config.yaml \
    --models M1=Qwen/Qwen2.5-3B-Instruct M2=meta-llama/Llama-3.2-3B-Instruct \
    --algorithm gspo_sapo_hybrid \
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
colab_reason_v1/
├── configs/
│   ├── base_config.yaml           # Base configuration
│   └── experiments/               # Experiment-specific configs
│       ├── pairwise_qwen_llama.yaml
│       ├── pairwise_qwen_family.yaml
│       ├── quad_model.yaml
│       └── all_pairs.yaml
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
│   │   └── combined_reward.py     # Combined reward function
│   ├── trainers/                  # Training logic
│   │   ├── base_trainer.py
│   │   ├── collab_trainer.py      # Main collaborative trainer
│   │   ├── micro_rounds.py        # Micro-round A/B logic
│   │   └── buddy_buffer.py        # Cross-teaching buffer
│   └── data/                      # Data loading
│       ├── dataset.py
│       └── preprocessing.py
├── scripts/
│   ├── train.py                   # Main training script
│   └── run_all_experiments.py     # Batch experiment runner
├── evaluation/
│   └── evaluate.py                # Evaluation script
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

### Combined Reward
```
R(τ) = w_explt × R_exploit + w_exp × R_explore + w_cross × R_cross + rescue_bonus
```

## Evaluation

```bash
# Evaluate a checkpoint
python evaluation/evaluate.py \
    --checkpoint outputs/experiment_name/checkpoint-final \
    --dataset gsm8k \
    --split test \
    --output results.json
```

### Metrics
- **Accuracy**: Per-model and combined (any model correct)
- **Rescue Rate**: P[B correct | A failed]
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
```

## References

- **GRPO**: DeepSeek-R1 (2024)
- **GSPO**: [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) - Qwen Team
- **SAPO**: [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347) - Qwen Team

## License

MIT License
