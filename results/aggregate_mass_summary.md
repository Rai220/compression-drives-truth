# Aggregate probability mass: correct mode vs the whole false support

Run 2026-07-27, in response to the NeurIPS meta-review request to rule out
probability-mass splitting ("evaluating the aggregate probability mass of the
incorrect modes vs. the correct mode") and Reviewer rFTw's identification concern.

## What is measured

Pair accuracy compares the correct completion against **one** sampled false
completion. A model that merely reproduces training frequencies assigns 0.5 to
the correct completion and 0.5/N to each of N false ones, so it wins every
pairwise comparison for N ≥ 2 without preferring the correct answer system.

Summing over the entire false support removes that:

    share = P(correct) / (P(correct) + Σ_i P(false_i))

* Frequency matching / probability-mass splitting → **share = 0.50 for every
  condition, every N, every model size.**
* A preference for the more compressible answer system → share > 0.50.

The false support is the exact set of incorrect derivations the training
generator can produce for that problem (`analysis/gen_random_support.py` mirrors
`data/generate_math.py`; `analysis/gen_multirule_support.py` mirrors
`data/generate_math_multirule.py` and reproduces the published paired test
problems exactly). Log-probabilities are summed over completion tokens, i.e.
true log P(string | prompt), not the length-normalised score used for pair
accuracy.

Evaluation only: existing checkpoints, no retraining.

## Core experiment — random vs coherent errors

4 seeds (42–45), mean ± std. n = 1101 problems (random), 915 (coherent).

| size | errors | false modes | E[pair acc] | **aggregate share** | equal-length subset |
|------|--------|------------:|------------:|--------------------:|--------------------:|
| tiny (3.5M) | random   | 5.51 | 76.3 ± 1.5 | **50.2 ± 0.7** | 53.2 ± 1.1 |
| tiny        | coherent | 1.00 | 46.7 ± 2.3 | 47.3 ± 0.7 | 49.3 ± 0.7 |
| small (12M) | random   | 5.51 | 81.4 ± 0.5 | **57.6 ± 0.2** | 57.9 ± 0.4 |
| small       | coherent | 1.00 | 50.7 ± 3.1 | 49.2 ± 0.4 | 49.5 ± 0.4 |
| medium (26M)| random   | 5.51 | 81.6 ± 0.7 | **59.0 ± 0.3** | 57.8 ± 0.2 |
| medium      | coherent | 1.00 | 54.4 ± 2.6 | 50.0 ± 0.4 | 49.9 ± 0.4 |
| large (86M) | random   | 5.51 | 83.4 ± 1.1 | **59.4 ± 0.6** | 58.5 ± 0.7 |
| large       | coherent | 1.00 | 54.2 ± 1.4 | 50.1 ± 0.1 | 49.7 ± 0.2 |

Per-seed bootstrap 95% CIs for the random condition lie entirely above 0.5 at
small, medium and large (e.g. large: [58.5,60.1] [59.0,60.7] [59.2,60.9]
[57.6,59.4]); every coherent CI contains or sits at 0.5. At tiny the random CIs
straddle 0.5.

**Reading.** The excess above 0.5 in the random condition cannot be produced by
frequency matching, and it is absent under coherent errors — where the false
support is a single mode and splitting is impossible by construction. It grows
with model size (50 → 58 → 59 → 59), a trend calibration predicts to be flat.

At 3.5M parameters the aggregate share is at chance while pair accuracy is 76%:
at that scale the pair-accuracy signal *is* splitting. The compression
preference appears from ~12M upward.

## Length control

False completions are on average **shorter** than the correct one
(−0.36 tokens, random; −0.71, coherent), so the length asymmetry works against
the reported effect rather than producing it. Restricting to the 359 problems
where every completion has identical token length leaves the numbers unchanged
(column 6 above).

## Multi-rule experiment

tiny, 4 seeds. N = 1 is the coherent baseline.

| N | distinct false modes | E[pair acc] | **aggregate share** |
|---|---------------------:|------------:|--------------------:|
| 1  | 1.00 | 47.0 ± 2.2 | 47.9 ± 0.7 |
| 2  | 2.00 | 79.1 ± 1.1 | 47.8 ± 0.4 |
| 3  | 2.91 | 83.1 ± 0.4 | 41.8 ± 0.6 |
| 5  | 4.26 | 86.1 ± 0.5 | 41.9 ± 0.5 |
| 10 | 8.52 | 90.1 ± 0.3 | 38.7 ± 1.0 |

**Reading.** Pair accuracy reproduces the published 47 → 78 → … → 88 pattern,
but the aggregate share does not rise at all — it is flat from N=1 to N=2 and
declines beyond. The N=1 → 2 jump is probability-mass splitting, as Reviewer
rFTw argued. The multi-rule experiment does not identify a selector-complexity
cost and should not be presented as doing so.

(Note that the declining share at large N is partly a property of the measure:
summing more neighbouring strings collects more of the model's diffuse
uncertainty. The safe conclusion is the absence of an increase, not the
presence of a decrease.)

## Per-type breakdown, random condition (aggregate share)

| type | false modes | tiny | small | medium | large |
|------|------------:|-----:|------:|-------:|------:|
| equation   | 1.42  | 49.9 | 51.2 | 51.3 | 51.7 |
| algebra    | 2.86  | 61.2 | 61.8 | 61.7 | 61.8 |
| derivative | 4.39  | 45.6 | 53.8 | 55.2 | 55.4 |
| arithmetic | 17.01 | 40.9 | 65.8 | 71.4 | 72.6 |

Equation problems have an almost degenerate false support (the training
generator produces at most two incorrect variants, and the wrong step is the
same one every time), and they show almost no excess — consistent with the
claim that the effect tracks the compressibility gap rather than the mere act
of summing.

## Reproduce

```bash
.venv/bin/python analysis/gen_random_support.py --n 1200 --seed 999 \
  --output data/corpus/support_random.jsonl
.venv/bin/python analysis/gen_multirule_support.py --n 1000 --seed 999 \
  --n-rules 2 --output data/corpus/support_multirule_2.jsonl

.venv/bin/python analysis/eval_aggregate_mass.py \
  --model-size large \
  --weights results/mixed_50_50_large_seed42/model_final.npz \
  --tokenizer results/mixed_50_50_large_seed42/tokenizer.json \
  --support data/corpus/support_random.jsonl \
  --output results/mixed_50_50_large_seed42/aggregate_mass_random.json
```

Per-model outputs: `results/*/aggregate_mass_{random,coherent,n<N>}.json`.

## Open issue found while doing this

`data/generate_paired_test.py` at seed 999 no longer reproduces the committed
`data/corpus/test_paired_random.jsonl` (the file matches its manifest hash, but
the current generator code produces a different draw — the multi-rule test files
still reproduce exactly). The shipped evaluation artefacts are unaffected;
only regeneration from the documented command diverges.
