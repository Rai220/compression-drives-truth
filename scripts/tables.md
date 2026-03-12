## Paired Random Proportions

| Proportion | Pair accuracy | Avg ΔLoss (paired) | Seed directions | n pairs |
|---|---|---|---|---|
| 50/50 | 83.1% +/- 2.4% | +0.0480 +/- 0.0012 | 4/4 > 50% | 4951-4951 |
| 40/60 | 79.0% +/- 1.7% | +0.0434 +/- 0.0010 | 4/4 > 50% | 4951-4951 |
| 30/70 | 74.7% +/- 1.3% | +0.0356 +/- 0.0018 | 4/4 > 50% | 4951-4951 |
| 20/80 | 69.3% +/- 1.1% | +0.0285 +/- 0.0015 | 4/4 > 50% | 4951-4951 |
| 10/90 | 66.7% +/- 0.4% | +0.0169 +/- 0.0022 | 4/4 > 50% | 4951-4951 |

## Corpus-Level Random Proportions (Legacy Window Estimate)

| Proportion | Loss correct | Loss incorrect | ΔLoss | 95% CI | Seeds -> correct |
|---|---|---|---|---|---|
| 100/0 | 0.1313 | 0.2028 | +0.0715 | N/A | 1/1 |
| 50/50 | 0.1384 +/- 0.0009 | 0.1499 +/- 0.0008 | +0.0115 +/- 0.0002 | [+0.0113, +0.0116] | 4/4 |
| 40/60 | 0.1403 +/- 0.0006 | 0.1492 +/- 0.0003 | +0.0089 +/- 0.0003 | [+0.0087, +0.0092] | 4/4 |
| 30/70 | 0.1422 +/- 0.0009 | 0.1486 +/- 0.0004 | +0.0064 +/- 0.0006 | [+0.0060, +0.0069] | 4/4 |
| 20/80 | 0.1455 +/- 0.0007 | 0.1487 +/- 0.0006 | +0.0033 +/- 0.0002 | [+0.0031, +0.0034] | 4/4 |
| 10/90 | 0.1503 +/- 0.0003 | 0.1487 +/- 0.0001 | -0.0016 +/- 0.0003 | [-0.0019, -0.0013] | 0/4 |

## Deterministic Full-Test Robustness Checks

| Condition | Loss correct | Loss incorrect | ΔLoss | Seeds -> correct |
|---|---|---|---|---|
| mixed 50/50 | 0.1089 +/- 0.0011 | 0.1246 +/- 0.0007 | +0.0157 +/- 0.0004 | 4/4 |
| mixed 10/90 | 0.1208 +/- 0.0004 | 0.1233 +/- 0.0002 | +0.0025 +/- 0.0003 | 4/4 |
| coherent 50/50 | 0.1086 +/- 0.0006 | 0.1077 +/- 0.0006 | -0.0008 +/- 0.0003 | 0/4 |

## Coherence Spectrum at 50/50 (Paired)

| Error type | Pair accuracy | Avg ΔLoss (paired) | Seed directions |
|---|---|---|---|
| Random | 83.1% +/- 2.4% | +0.0480 +/- 0.0012 | 4/4 > 50% |
| Contradictory | 49.0% +/- 1.2% | +0.0003 +/- 0.0006 | 1/4 > 50% |
| Coherent | 47.2% +/- 2.8% | -0.0018 +/- 0.0006 | 1/4 > 50% |

## Paired Robustness Checks

| Condition | Primary accuracy | Primary ΔLoss | Sum-NLL accuracy | Sum-NLL ΔLoss | Matched-length accuracy | Matched-length ΔLoss |
|---|---|---|---|---|---|---|
| math random 50/50 | 83.1% +/- 2.4% | +0.0480 +/- 0.0012 | 82.7% +/- 2.1% | +3.5634 +/- 0.1060 | 82.7% +/- 2.1% | +0.0481 +/- 0.0012 |
| math random 10/90 | 66.7% +/- 0.4% | +0.0169 +/- 0.0022 | 66.4% +/- 0.5% | +1.4104 +/- 0.1506 | 66.4% +/- 0.5% | +0.0162 +/- 0.0022 |
| math coherent 50/50 | 47.2% +/- 2.8% | -0.0018 +/- 0.0006 | 46.2% +/- 3.0% | -0.1737 +/- 0.0314 | 46.2% +/- 3.0% | -0.0024 +/- 0.0006 |
| world random 50/50 | 57.7% +/- 1.9% | +0.0340 +/- 0.0105 | 75.7% +/- 2.4% | +0.8015 +/- 0.6513 | 75.7% +/- 2.4% | +0.0112 +/- 0.0109 |
| world coherent 50/50 | 46.6% +/- 2.1% | +0.0192 +/- 0.0137 | 46.8% +/- 3.0% | -0.9962 +/- 0.8667 | 46.9% +/- 3.0% | -0.0159 +/- 0.0140 |

## Multi-Rule Matched Evaluation

| N rules | Pair accuracy | Avg ΔLoss (paired) | Evaluation family |
|---|---|---|---|
| 1 | see `results_manifest.md` | see `results_manifest.md` | matched coherent baseline stored separately |
| 2 | 77.6% +/- 1.5% | +0.0152 +/- 0.0007 | matched (`eval_paired_matched.json`) |
| 3 | 82.8% +/- 0.1% | +0.0213 +/- 0.0006 | matched (`eval_paired_matched.json`) |
| 5 | 84.8% +/- 0.6% | +0.0293 +/- 0.0009 | matched (`eval_paired_matched.json`) |
| 10 | 88.3% +/- 0.8% | +0.0440 +/- 0.0010 | matched (`eval_paired_matched.json`) |

## Scaling Coverage

| Condition | tiny | small | medium | large |
|---|---|---|---|---|
| random 50/50 paired seeds | 4 | 4 | 4 | 4 |
| coherent 50/50 paired seeds | 4 | 4 | 4 | 4 |
