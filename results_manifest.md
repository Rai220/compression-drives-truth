# Results Manifest

This file records which released artifacts back the revised manuscript and which gaps remain.

Companion manifests:

- [`claims_manifest.md`](claims_manifest.md) - which claims are safe, qualified, or removed
- [`data/eval_inputs_manifest.md`](data/eval_inputs_manifest.md) - evaluation input provenance and checksums

## Scaling Coverage

Paired-evaluation coverage for the main random/coherent 50/50 scaling runs:

| Size | Random `eval_paired.json` | Coherent `eval_paired.json` | Notes |
|---|---:|---:|---|
| `tiny` | `4/4` available (`seed42` stored under legacy path `mixed_50_50_tiny/`) | `4/4` available | complete |
| `small` | `4/4` available | `4/4` available | complete |
| `medium` | `4/4` available | `4/4` available | complete |
| `large` | `4/4` available | `4/4` available | complete |

The current public release therefore supports full random and coherent fixed-step size trends across `tiny`, `small`, `medium`, and `large`. These size comparisons should still be described as fixed-step results rather than as compute-matched scaling laws.

## Experiment 5: Multi-Rule Math

Legacy public files:

- `results/multirule_*_50_50_tiny_seed*/eval_paired.json`

These legacy files evaluate multi-rule-trained models on `data/corpus/test_paired_random.jsonl`. They measure transfer to the random paired benchmark and should **not** be interpreted as matched `correct` vs `multi-rule incorrect` evaluation.

Matched artifacts added in this revision:

- `data/corpus/test_paired_multirule_1.jsonl`
- `data/corpus/test_paired_multirule_2.jsonl`
- `data/corpus/test_paired_multirule_3.jsonl`
- `data/corpus/test_paired_multirule_5.jsonl`
- `data/corpus/test_paired_multirule_10.jsonl`
- `results/multirule_*_50_50_tiny_seed*/eval_paired_matched.json`
- `results/coherent_50_50_tiny_seed*/eval_paired_multirule_n1.json`

Matched paired-evaluation summary across 4 seeds:

| N rules | Artifact family | Mean pair accuracy | Mean delta |
|---:|---|---:|---:|
| `1` | coherent baseline on matched `N=1` test | `46.6%` | `-0.0019` |
| `2` | matched multi-rule test | `77.6%` | `+0.0152` |
| `3` | matched multi-rule test | `82.8%` | `+0.0213` |
| `5` | matched multi-rule test | `84.8%` | `+0.0293` |
| `10` | matched multi-rule test | `88.3%` | `+0.0440` |

Interpretation: after rebuilding Experiment 5 on matched paired tests, the effect remains substantial but is more graded than the legacy `49% -> 87%` narrative. Only the matched files above should be used for the revised paper and revised figure scripts.

## Claim Status

| Claim | Publicly supported now | Notes |
|---|---|---|
| Random 50/50 paired preference for correct completions | yes | backed by released paired JSONs |
| Random 10/90 paired preference remains positive | yes | use paired outputs, not corpus-level DLoss alone |
| Random fixed-step size trend | yes | public artifacts complete across sizes |
| Coherent scaling remains near chance across released sizes | yes | use fixed-step wording; paired accuracy stays close to 50% and mean paired DLoss stays near zero |
| Multi-rule graded rise under matched evaluation | yes | rebuilt in this revision |
| Legacy `49% -> 87%` multi-rule jump | no | legacy experiment mixed train and test distributions |
| Chained-task decline with size | partial | large uses only 2 seeds |
| Broad generalization from synthetic corpora to truthfulness in the wild | no | should remain outside the revised main claim |

## Deterministic Full-Test Corpus Checks

New robustness artifacts added for the revision:

- `results/baseline_correct_tiny/eval_perplexity_full.json`
- `results/mixed_50_50_tiny*/eval_perplexity_full.json`
- `results/mixed_10_90_tiny_seed*/eval_perplexity_full.json`
- `results/coherent_50_50_tiny_seed*/eval_perplexity_full.json`

Summary across 4 seeds:

| Condition | Mean deterministic DLoss | Seeds -> correct | Interpretation |
|---|---:|---:|---|
| random `50/50` | `+0.0157` | `4/4` | agrees with paired and legacy corpus signs |
| random `10/90` | `+0.0025` | `4/4` | removes the legacy sign inversion |
| coherent `50/50` | `-0.0008` | `0/4` | agrees with the paired coherent result |

## Paired Robustness Checks

Auxiliary paired metrics added in this revision:

- `sum_nll` over completion tokens
- `length_matched_mean_nll` on the shared minimum completion length

Key summary for the central math conditions:

| Condition | Primary paired result | `sum_nll` | `length_matched_mean_nll` |
|---|---|---|---|
| random `50/50` | `83.1%`, `ΔLoss = +0.0480` | `82.7%`, `ΔLoss = +3.5634` | `82.7%`, `ΔLoss = +0.0481` |
| random `10/90` | `66.7%`, `ΔLoss = +0.0169` | `66.4%`, `ΔLoss = +1.4104` | `66.4%`, `ΔLoss = +0.0162` |
| coherent `50/50` | `47.2%`, `ΔLoss = -0.0018` | `46.2%`, `ΔLoss = -0.1737` | `46.2%`, `ΔLoss = -0.0024` |

Interpretation: in the central math conditions, the primary sign pattern is preserved by both auxiliary paired variants. This reduces the residual concern that the main paired result is an artifact of completion-length asymmetry.

## Recommended Use In The Revised Paper

- Treat paired evaluation as primary evidence.
- Treat corpus-level DLoss as a secondary diagnostic.
- Do not cite legacy `multirule/*/eval_paired.json` as matched multi-rule evidence.
- Cite coherent scaling across all released sizes only with explicit fixed-step wording.
