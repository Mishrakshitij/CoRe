# CoRe: Collaborative Reasoning via Cross Teaching

**Kshitij Mishra, Mirat Aubakirov, Martin Takac, Nils Lukas, and Salem Lahlou**

[Paper (arXiv v2)](https://arxiv.org/abs/2601.21600v2) · [PDF](https://arxiv.org/pdf/2601.21600v2) · [Reproducibility notes](docs/reproducibility.md)

CoRe trains language models together: each first attempts a problem independently, then samples again with an optional hint from a successful peer. The objective combines correctness, diversity, and a bonus for successful rescue. Evaluation uses cold prompts; no peer hints are needed at test time.

This repository now includes a compact, inspectable reference implementation in `core/`, with runnable training and evaluation commands, validated configuration, and CPU regression tests. The older `src/` trainers and YAML experiments remain available as historical variants. This update has not reproduced the paper's benchmark tables.

## Quick start

Use Python 3.10 or newer. Run commands from the repository root.

```bash
git clone https://github.com/Mishrakshitij/CoRe.git
cd CoRe
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-paper.txt
```

Install a PyTorch build compatible with your GPU/CUDA environment. The reference trainer uses LoRA and explicit devices, one device per model; it does not use the legacy distributed or 4-bit training backends. `requirements-paper.txt` contains compatibility ranges, not a recovered original environment lockfile.

For CPU-only code checks, install just NumPy:

```bash
python -m pip install 'numpy>=1.26,<3'
python -m unittest discover -s tests -v
python -m core.cli --config configs/paper_qwen.json --data examples/train.jsonl --validate-only
python -m core.evaluation examples/predictions.jsonl --k 2
```

The included examples are synthetic fixtures for checking the interface, not training data or benchmark results. Tensor tests skip explicitly when PyTorch is unavailable.

## Training

Provide your selected training split as JSONL, with one record per example:

```json
{"id":"example-001","question":"What is 7 + 5?","answer":"12"}
```

Use stable, unique IDs and final answers, not full gold solutions. For multiple-choice tasks, include the options in `question` and use a consistent label in `answer`. Keep a separate held-out manifest for evaluation.

```bash
python -m core.cli \
  --config configs/paper_qwen.json \
  --data data/train.jsonl \
  --output-dir outputs/core-qwen-run1
```

Edit model IDs, revisions, and devices in the JSON configuration for your hardware. The default Qwen pair uses `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen3-4B-Instruct-2507`; the dated Qwen3 identifier is an explicit reference choice for the paper's shortened name.

`configs/paper_reasoning.json` provides the Phi-4-mini-reasoning / Ministral-3-3B-Reasoning pair. The shared loader handles Ministral's conditional-generation architecture and native tokenizer. This path requires Transformers 5 and `mistral-common`, included in the reference requirements; backbone execution still needs GPU validation.

The training loop retains the exact prompt and generated token IDs for every rollout. It collects both rounds and freezes all old/reference log probabilities before either model updates. Advantages are normalized across all models and both rounds for the same problem. The reference trainer implements GRPO; it rejects unsupported optimizer names instead of silently substituting another objective.

| Setting | Reference preset |
| --- | --- |
| Cold samples per model, K | 2 |
| Round B samples per model, K' | 1 |
| Probability of supplying a hint | 0.75 |
| Hint budget | 1536 tokens |
| Training completion budget | 3072 tokens |
| Round B loss weight | 0.8 |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 1e-5 |
| Semantic / structural distance weights | 0.6 / 0.4 |

Some paper descriptions conflict or leave details unspecified, including hint probability, teacher ranking, partial credit, and dataset splits. The preset and [decision notes](docs/reproducibility.md) identify the choices made here. The literal exploration equation can yield all-zero diversity rewards; diagnostics expose that behavior.

Each run saves its configuration, data manifest, training metrics, and model adapters under the output directory. Optional rollout logging retains the actual prompts and samples. Adapter exports support inference; they are not full optimizer/RNG resume checkpoints. Existing nonempty run directories are rejected.

## Cold evaluation

Generate two cold samples per model from the exported adapters:

```bash
python -m core.generate \
  --config outputs/core-qwen-run1/config.json \
  --checkpoint-dir outputs/core-qwen-run1/final \
  --data data/test.jsonl \
  --k 2 --max-new-tokens 4096 \
  --output outputs/core-qwen-pass2.jsonl

python -m core.evaluation outputs/core-qwen-pass2.jsonl \
  --k 2 --output outputs/core-qwen-pass2-metrics.json
```

Omit `--checkpoint-dir` to evaluate the base models. For a separate greedy Pass@1 run, use `--k 1 --greedy` and a new output filename. `--validate-only` checks inputs without loading weights. Generation stores tokenizer counts and a metadata sidecar with decoding settings and completion status.

The scorer accepts JSONL with `example_id`, `model`, `gold`, and a list of `completions`; optional `token_counts` must align with every completion. Rows may arrive in any order. IDs must match across all models.

- **Individual Pass@K:** any of a model's first K samples is correct.
- **Oracle team Pass@K:** any collaborator succeeds; two models at K=2 use four samples per example.
- **Majority-vote accuracy:** a separate selector without gold access, with deterministic ties.
- **Collaboration gain:** oracle team score minus the best individual score.

Oracle team performance is an upper bound using gold correctness, not deployable answer selection. Answer normalization is deterministic but is not a general symbolic mathematics verifier. Preserve raw outputs for dataset-specific rescoring.

## Code map

| Path | Purpose |
| --- | --- |
| `core/parsing.py` | Shared answer extraction and numeric comparison |
| `core/rewards.py` | Paper reward components, hybrid distances, grouped advantages |
| `core/protocol.py` | Teacher selection, hint sanitization, prompts |
| `core/trainer.py` | Two-round orchestration, LoRA policy, GRPO updates |
| `core/cli.py` | Configuration/data validation and training entry point |
| `core/generate.py`, `core/evaluation.py` | Cold generation and aligned team scoring |
| `configs/paper_qwen.json` | Named reference configuration |
| `tests/` | Numerical, protocol, parser, and evaluation regressions |
| `src/`, older YAML configs | Preserved experimental variants |

## Citation

```bibtex
@article{mishra2026core,
  title={CORE: Collaborative Reasoning via Cross Teaching},
  author={Mishra, Kshitij and Aubakirov, Mirat and Takac, Martin and Lukas, Nils and Lahlou, Salem},
  journal={arXiv preprint arXiv:2601.21600},
  year={2026},
  doi={10.48550/arXiv.2601.21600},
  url={https://arxiv.org/abs/2601.21600v2}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
