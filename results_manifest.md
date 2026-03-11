# Results Manifest

This file records which public artifacts currently back the main claims in the repository and which gaps remain.

## Scaling Coverage

Paired-evaluation coverage for the main random/coherent 50/50 scaling runs:

| Size | Random `eval_paired.json` | Coherent `eval_paired.json` | Notes |
|---|---:|---:|---|
| `tiny` | `4/4` available (`seed42` stored under legacy path `mixed_50_50_tiny/`) | `4/4` available | complete |
| `small` | `4/4` available | `4/4` available | complete |
| `medium` | `4/4` available | `0/4` available | coherent artifacts missing |
| `large` | `4/4` available | `1/4` available (`seed42` only) | coherent artifacts incomplete |

The current public release therefore supports a full random fixed-step size trend, but only partial public replication for coherent scaling.

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

Interpretation: after rebuilding Experiment 5 on matched paired tests, the effect remains substantial but is more graded than the legacy `49% -> 87%` narrative.

## Claim Status

| Claim | Publicly supported now | Notes |
|---|---|---|
| Random 50/50 paired truth preference | yes | backed by released paired JSONs |
| Random fixed-step size trend | yes | public artifacts complete across sizes |
| Coherent scaling flatline across all sizes | partial | medium missing, large only `seed42` |
| Multi-rule transition under matched evaluation | yes | rebuilt in this revision |
| Legacy `49% -> 87%` multi-rule jump | no | legacy experiment mixed train and test distributions |
| Chained-task decline with size | partial | large uses only 2 seeds |
