# Claims Manifest

This file records which claims are safe for the revised manuscript, which ones must be qualified, and which legacy narratives should not be reused.

## Main-Text Safe Claims

| Claim | Status | Artifact basis | Notes |
|---|---|---|---|
| Random errors produce strong paired preference for correct completions at `50/50` | safe | `results/mixed_50_50_tiny*/eval_paired.json` | core result |
| Random errors retain paired preference even at `10/90` | safe | `results/mixed_10_90_tiny_seed*/eval_paired.json` | use paired metric, not corpus-level DLoss |
| Coherent errors remove or reverse paired preference at `50/50` | safe | `results/coherent_50_50_tiny_seed*/eval_paired.json` | central contrast with random errors |
| Key math paired results are stable under sum-based and length-matched robustness variants | safe | `results/mixed_50_50_tiny*/eval_paired.json`, `results/mixed_10_90_tiny_seed*/eval_paired.json`, `results/coherent_50_50_tiny_seed*/eval_paired.json` | applies to the central math conditions only |
| Corpus-level and paired metrics can diverge | safe | `mixed_10_90_*`, `condC/D/E_*`, appendix world artifacts | important negative result |
| Multi-rule matched evaluation yields a graded rise from coherent `N=1` to `N=10` | safe | `results/coherent_50_50_tiny_seed*/eval_paired_multirule_n1.json`, `results/multirule_*_50_50_tiny_seed*/eval_paired_matched.json` | use matched evaluation only |
| Chained verification restores preference at tiny scale | safe | `results/chained_50_50_tiny_seed*/eval_paired.json` and coherent control | claim should stay at tiny scale unless explicitly qualified |
| Synthetic world random-error preference is weaker than in math | safe | `results/world_random_50_50_tiny_seed*/eval_paired.json` | appendix / secondary evidence |
| In the released fixed-step scaling runs, coherent falsehood remains near chance across `3.5M`--`86M` | safe | `results/coherent_50_50_{tiny,small,medium,large}_seed*/eval_paired.json` | keep the fixed-step qualifier |

## Claims Requiring Qualification

| Claim | Status | Why it must be qualified | Allowed wording |
|---|---|---|---|
| Random-error preference strengthens with size | qualified | fixed-step training, not compute-matched | \"in the available fixed-step runs\" |
| Chained tasks show a declining size trend | qualified | large uses only 2 seeds | \"preliminary fixed-step trend\" |
| Synthetic-world results generalize beyond mathematics | qualified | weaker effect size, appendix-only evidence | \"suggestive but exploratory\" |
| Deterministic full-test corpus-level eval matches the original narrative | qualified | robustness check, not the primary paper result | \"supports / is consistent with\" |

## Claims To Remove Or Avoid

| Legacy claim | Status | Why |
|---|---|---|
| `training more than 160 small character-level transformers on mathematical corpora` | avoid | the `160+` total includes appendix domains and non-math conditions |
| `3 sizes x 2 conditions x 4 seeds = 22 models` | remove | arithmetic error; if retained, the count must be corrected or the section rewritten around released coverage |
| Legacy multi-rule `49% -> 87%` jump | remove | based on mismatched evaluation family (`eval_paired.json` vs random benchmark) |
| Broad statements that compression explains truthfulness in general | remove | the evidence is restricted to synthetic corpora and fixed-step training |
| Strong inverse-scaling rhetoric for chained tasks | remove | evidence is too thin for a law-like claim |

## Preferred Framing For The Revised Paper

- Models compress **text**, not reality.
- Paired evaluation is the primary metric because it holds the prompt fixed.
- Corpus-level DLoss is a secondary diagnostic and can be confounded by format, length, and frequency effects.
- The safest main conclusion is about **compressibility and internal consistency**, not truth in the abstract.
