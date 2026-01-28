# CoRe: Collaborative Reasoning via Cross Teaching

CoRe trains multiple language models to solve reasoning problems collaboratively during training. It uses a two-round micro-round protocol that turns peer success into a learning signal, while explicitly rewarding diverse reasoning traces to reduce correlated errors.

## Overview

CoRe combines:
- **Micro-round cross-teaching**: Round A cold sampling, Round B contexted rescue with peer hints.
- **Explore–exploit rewards**: Correctness for accuracy, DPP-lite diversity to reduce overlap.
- **Rescue bonus**: Extra reward when a failed model recovers using a peer hint.
- **Optional rewards**: Think-diversity, format adherence, exact match, length shaping, trace-accuracy.

### High-level flow
1. **Round A (cold)**: Each model generates `K` traces per question.
2. **Hint selection**: If any model is correct, extract a hint (compressed or full trace).
3. **Round B (contexted)**: Models regenerate with optional hint dropout.
4. **Policy update**: Combine rewards and update the policy with GRPO/GSPO/SAPO.

### Multi-strategy prompt format
For multi-strategy runs, each response includes per-strategy outcomes and a final answer:
```
<strategy id="1">
<approach>...</approach>
<reasoning>...</reasoning>
<strategy_id_outcome>...</strategy_id_outcome>
</strategy>

<strategy id="2">...

<final_answer>...</final_answer>
```
The tag name is configurable via `prompting.strategy_outcome_tag` (default `result`).

## Repository structure

- `src/` core training logic: trainers, rewards, data, prompts
- `configs/` YAML configs for datasets and model pairs
- `scripts/` training entrypoints
- `evaluation/` evaluation scripts (vLLM and torch-based)
- `outputs/` checkpoints and experiment artifacts
- `train_logs/` per-run logs (created automatically)
- `evaluation_results/` JSON evaluation outputs

## Setup

```bash
pip install -r requirements.txt
```

**Mistral-3 reasoning models** require `transformers>=5` and `mistral-common`. Use a dedicated env if needed.

## Training

Fast training (recommended):
```bash
CUDA_VISIBLE_DEVICES=0,2,3,6 \
  python scripts/train_fast.py \
  --config configs/grpo_gpqa_main_phi4mini_ministral3_3b_fp16_4gpu_strategy.yaml \
  --prompt-template auto
```

Key config knobs:
```yaml
prompting:
  multi_strategy: true
  template_type: auto
  context_template: auto
  dataset_prompt_source: template
  strategy_outcome_tag: strategy_id_outcome

rewards:
  exploit_answer_source: strategy_outcome
  correctness_answer_source: final_answer
  explore_scope: strategies
  use_correctness_reward: true
  use_exact_match_reward: true
  use_format_reward: true
  use_length_reward: true
```

Logs are written to `train_logs/<experiment_name>/training_<timestamp>.log`.

## Evaluation

### vLLM multi-strategy evaluation
```bash
CUDA_VISIBLE_DEVICES=3 \
  python evaluation/evaluate_vllm_multi_strategy.py \
  --checkpoint /path/to/checkpoint \
  --model M1 \
  --dataset gpqa_main --split test \
  --batch-size 16 --num-traces 2 \
  --max-model-len 10244 --max-tokens 8192 \
  --dataset-prompt-source template \
  --strategy-outcome-tag strategy_id_outcome \
  --scoring-mode final_answer
```

### Torch-based evaluation (no vLLM)
```bash
python evaluation/evaluate.py \
  --config configs/base_config.yaml \
  --checkpoint /path/to/checkpoint \
  --dataset gpqa_main \
  --prompt-style multi-strategy
```
Ensure the config includes:
```yaml
prompting:
  dataset_prompt_source: template
  strategy_outcome_tag: strategy_id_outcome
rewards:
  use_correctness_reward: true
  correctness_answer_source: final_answer
```

## Branches

- `main`: stable baseline (legacy XML prompt schema, `<result>` outcomes).
- `feature/strategy-outcome-prompts`: new multi-strategy prompt templates, `<strategy_id_outcome>` tags, strategy-level exploration, and extended reward controls.

## Notes

- GPQA is gated on Hugging Face. Run `huggingface-cli login` and request access before training/evaluating.
- Use `prompting.dataset_prompt_source: legacy_xml` and `prompting.strategy_outcome_tag: result` if you need backward compatibility with older checkpoints.
