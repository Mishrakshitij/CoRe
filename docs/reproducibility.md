# Reproducibility and implementation decisions

The reference implementation targets [CoRe, arXiv:2601.21600v2](https://arxiv.org/abs/2601.21600v2), revised August 3, 2026. It is a maintainable implementation of the described protocol, not an archive of the original experiment environment or checkpoints.

## Scope

The new `core/` path is separate from the older experimental trainers in `src/`. The latter contain additional reward variants, distillation, distributed backends, and model-specific configurations. They remain available for historical experiments; their results and settings must not be attributed automatically to the paper reference path.

Training requires caller-supplied JSONL data with stable IDs. No guessed split is labeled as an original paper split. Keep dataset revision, source split, selection seed, exact IDs, prompt template, model/tokenizer revisions, dependency versions, generation settings, and run configuration with each experiment. Use disjoint train and evaluation IDs and inspect near-duplicate questions before interpreting benchmark results.

## Paper decisions that need to remain visible

| Topic | Evidence in the paper | Reference interpretation |
| --- | --- | --- |
| Hint probability | Section 5 gives 0.75; Appendix A Table 4 gives 0.5 and uses dropout terminology | Config names the probability of **using** a hint explicitly; default 0.75 |
| Teacher ranking | Section 4 selects highest total reward; Appendix A describes exploitation reward | Total reward by default; selection policy is explicit |
| Partial credit | Equation 2 gives a bounded overlap score without a complete algorithm | The default and any callback must be recorded; changing it changes the reward |
| Operation signatures | Equation 13 specifies semantic and structural distance; exact regex rules are not fully listed | The committed regex rules are reference implementation choices |
| Exploration subset | Equation 3 assigns zero to selected members and rewards outsiders beyond the margin | The literal equation is preserved; zero exploration is possible and observable |
| Cross reward | Enabled in main experiments, late weight 0.1; gate and margin are incompletely specified | Config exposes the gate, margin, and weights rather than hiding constants |
| Model identity | Paper uses shortened model names | The config contains executable Hugging Face IDs; pin exact revisions for an experiment |
| AIME split | Abstract gives 792 training examples; Appendix I gives 770 | Supply and retain an explicit data manifest; neither count proves original split identity |
| Optimizer variants | GRPO, GSPO, SAPO appear in the paper and legacy code | Only implemented, tested reference objectives are accepted by the new CLI |

The named presets use normalized-token multiset F1 as the partial-credit rule with alpha 0.3. The gate threshold 0.1, cross margin 0.15, trace-accuracy weight 0.2, AdamW weight decay 0.01, warmup 0.1, two epochs, and the exploration schedule 0.2 to 0.05 are explicit reference/legacy choices where the paper does not identify a complete experimental recipe. Cross weight changes from 0 to 0.1 in the second epoch. The generic reward dataclass has conservative defaults that can differ from these named presets. For more than two collaborators, cross rewards average over partners; this is an explicit extension of the pairwise equation.

### Exploration reward can be identically zero

For a maximal greedy margin-separated subset below its size cap, every unselected trace lies within the margin of a selected trace. Equation 3 then gives every outsider zero reward, and selected members receive zero by definition. With two models and K=2, K'=1, only six traces are collected, below the reported cap of ten. Treat an all-zero exploration diagnostic as a property to examine, not as evidence that diversity is successfully optimized. A different selection rule or leave-one-out reward would be a method change and should be reported as an ablation.

### Teacher hints and answer handling

Explicit final-answer fields and answer lines are removed before a hint is supplied. The correct strategy can still contain arithmetic that reveals an answer; tag removal is not a proof of semantic answer secrecy. Empty hints do not qualify for rescue rewards. Round B is trained on the actual prompt that produced each completion, including whether the hint was dropped.

Answer comparison is a deterministic normalization utility, not a full symbolic mathematics verifier. Review dataset-specific scoring before MATH, AIME, or GPQA claims. Preserve raw completions and extracted answers so scoring can be audited.

## Evaluation interpretation

`python -m core.evaluation` scores cold completions by stable example ID. `--k 2` means two samples **per model**: a two-model team uses four candidate solutions per example. Oracle team Pass@K uses gold correctness to select success; it is not a deployable selector. Majority vote is reported separately, with deterministic tie handling. The evaluator rejects missing model/example pairs, conflicting gold labels, duplicates, insufficient samples, and hinted Round B predictions.

Pass@1 from the first stochastic sample is not identical to a separate greedy evaluation. Generate and label separate runs when comparing decoding policies. Token counts are reported only when supplied for every sample; word counts are not substituted for tokenizer counts. The included prediction fixture is synthetic and its token counts are illustrative.

## Validation boundary

The native trainer evaluates the GRPO surrogate using raw model likelihoods, while the default generation settings apply temperature and top-p sampling. This common training approximation does not make those raw likelihoods the exact truncated sampling distribution. Use the recorded likelihood mode and decoding settings when comparing implementations.

CPU tests cover parsing, numerical rewards, protocol invariants, and evaluation accounting. Optional tensor tests require PyTorch. This update does not include downloaded language-model weights, a GPU training run, restored original checkpoints, or newly reproduced paper tables. Running the code is a prerequisite to measuring performance, not evidence of matching the published results.
