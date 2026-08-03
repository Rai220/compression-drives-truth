# Held-out measurement on a memorisation-free corpus

Run 2026-07-28 on 8xH100. Raw per-run JSON in `results/scale_*/heldout.json`,
held-out evaluation inputs in `data/heldout_*_clean.jsonl`.

> **Corrected 2026-08-03.** This document describes the corpus and the run, and
> that part still stands. Its conclusions about the `keyed` condition do not: the
> family measurement run on 2026-08-01/02 (`results/scale2_keyed_*`,
> `results/scale_keyed_qwen3-1b_*/heldout_family.json`) shows that keyed is the
> same measurement as random once both are scored on a like-for-like support, and
> that the keyed-vs-coherent gap reported below compares two different null
> levels. The corrected reading, with the null for every measure derived from the
> corpus, is in `SCALING_LAW.md`. Sections below are annotated where they fail.

## Why this run exists

Every earlier aggregate-mass number was taken on **training** problems, because
that is what makes corpus frequencies exactly known. That construction has a
ceiling: once a model can memorise both strings it simply reproduces their
frequencies, so the measured preference decays toward chance as capacity grows.
We appeared to hit that ceiling on the 5k-problem denoising corpus, where the
realized-support measure peaks at 26M and falls away (80.3 at 26M, 77.6 at 86M
in the four-seed rerun, and lower still at 200M/420M in runs no longer on disk).
The family measure tells a different story on the same corpus — it rises
monotonically, 18.6 → 31.6 → 44.1 → 59.0 — so the "ceiling" was itself partly an
artefact of scoring against the single realised string. See `docs/mass_control.md`.

This corpus removes the ceiling:

* 500,000 problems per condition, 92 MB, **93.8% of texts distinct**, maximum
  repetition of any single text 11;
* training consumes 41M tokens, i.e. **0.44 epochs** — most problems are seen
  once or not at all;
* the held-out split is disjoint **by construction** (a hash of the problem
  parameters decides train vs held-out), then filtered against the training set
  string-by-string. Residual leakage: 0 of 2533 / 2523 / 2512.

## Conditions

All three corrupt one step of the derivation, at matched corruption load.

| condition | false modes per problem | error |
|---|---|---|
| `random` | 17.5 | a fresh draw from the error family each time |
| `keyed` | **1** | the same family, but the variant is a hash of the problem |
| `coherent` | **1** | one fixed wrong rule everywhere |

`keyed` was introduced to answer the identification objection, on the reasoning
that it holds the number of false modes at one, exactly like `coherent`, so
aggregate probability mass would be comparable and splitting impossible by
construction.

**That reasoning is wrong, and this is the central correction.** `mean_false_modes = 1`
is a property of the support *file*, not of the model's predictive distribution.
The variant is selected by `blake2b` of the problem parameters; on a held-out
problem the model cannot evaluate that hash, so its false mass necessarily
spreads over the whole 17.5-member family, and scoring against the one selected
string credits that spread to the correct completion. Measured on the family
support, keyed and random agree to about a point at every size (13.3/12.7 at
3.5M, 37.8/36.6 at 86M): keyed *is* random. The frequency-matching null for the
keyed realized measure is E[n/(n+1)] = 94.08, not 0.5, and every measured value
lies below it. `generate_scale.py`'s own `gen_heldout` docstring says as much.

## Results

GPT-2 family, 4 seeds at every cell, mean ± std, held-out problems. (The
original text said `keyed` xlarge had 2 seeds and quoted seeds 44+45 only; all
four landed shortly after and the values below are the four-seed ones.)

### Pair accuracy — correct completion against one false completion

| size | random | keyed | coherent |
|---|---|---|---|
| tiny 3.5M | 72.2 ± 1.9 | 73.0 ± 1.9 | 36.5 ± 1.3 |
| small 12M | 79.1 ± 0.2 | 79.4 ± 0.5 | 42.8 ± 2.3 |
| medium 26M | 83.1 ± 0.3 | 83.2 ± 0.5 | 44.7 ± 1.2 |
| large 86M | 89.5 ± 1.3 | 89.4 ± 1.2 | 54.3 ± 2.0 |
| xlarge 200M | 94.3 ± 0.6 | 91.3 ± 1.3 | 55.5 ± 1.9 |

`random` and `keyed` agree to within a point at every size below 200M. At the
time this was read as "making the error deterministic changes nothing of
substance"; the family measurement gives the stronger version — the two
conditions are indistinguishable because from the model's side they are the same
condition. The nulls for these three columns are 100.0, 100.0 and 38.7
respectively (see `SCALING_LAW.md` §1), so none of them is informative about
preference.

### Aggregate probability mass, `keyed` vs `coherent` — **retracted**

| size | keyed (realized) | coherent | printed gap |
|---|---|---|---|
| tiny 3.5M | 67.4 ± 1.5 | 37.6 ± 0.5 | +29.8 |
| small 12M | 74.1 ± 0.6 | 43.7 ± 1.9 | +30.5 |
| medium 26M | 77.9 ± 0.5 | 47.6 ± 0.5 | +30.3 |
| large 86M | 83.3 ± 0.8 | 51.9 ± 0.6 | +31.3 |
| xlarge 200M | 84.9 ± 0.9 | 51.9 ± 0.5 | +33.1 |

The numbers reproduce (the 200M row is corrected from seeds 44+45 to all four);
the reading does not. The keyed column has a null of 94.08 and the coherent
column a null of 50.00, so the "gap" is the distance between two different
indifference points, not an effect. Scored on a common support the sign
reverses: keyed-family minus coherent is −24.3 at 3.5M, −19.1 at 12M, −17.2 at
26M, −14.2 at 86M and −2.3 at 1B, i.e. keyed *below* coherent and narrowing.
This table must not be cited as a positive result.

## Length control, and a caveat on the coherent condition

Completion-length differences: `keyed` +0.07 tokens, `random` −0.03,
**`coherent` −1.39**. The coherent rule produces systematically shorter output,
and shorter strings carry more probability for reasons unrelated to the
hypothesis. On the subset where correct and incorrect completions have
identical token length:

| size | keyed (n=1918) | coherent (n=1255) |
|---|---|---|
| tiny | 69.0 | 49.4 |
| small | 76.7 | 51.6 |
| medium | 80.7 | 51.2 |
| large | 86.1 | 52.4 |
| xlarge | 87.6 | 51.9 |

The length control itself stands and is the durable finding of this run.
**`coherent` moves**: 37.6 → 49.4 at tiny, and the below-chance readings
disappear entirely — length-matched, the coherent condition sits at chance, not
below it. The "models prefer coherent falsehood" reading of the sub-50 numbers
is a length artefact and must not be claimed. The keyed column is unaffected by
length (67.4 → 69.0 at tiny) but is measured against the wrong null, so the
"keyed-vs-coherent gap survives at 20–36 points" sentence that stood here is
withdrawn.

## Reading (corrected 2026-08-03)

**Coherent falsehood is not filtered at any scale.** Length-matched, the
coherent condition reads 49.4 at 3.5M and 50.2 at 1B against an exact 50.0 null,
flat across 2.5 decades. One false mode means splitting is impossible, so this
half needs no further control and is the result to quote.

**Pair accuracy on this corpus is uninformative.** Its null is 100.0 for random
and keyed (a frequency-matching model wins every pair) and 38.7 for coherent
(decided by length). The 36–39 point "gap" is the distance between two different
nulls. The earlier sentence claiming the random-vs-coherent contrast "holds on
unseen problems and does not decay with scale" rested on it and is withdrawn.

**The 0.5 line does apply off the training set — for the family measure.**
Summing the whole enumerated family is invariant to how the false half is split,
so the frequency-matching null is exactly 0.5 for random, keyed-family and
coherent alike (band [0.500, 0.513] if the model applies the corpus marginal
instead of the per-problem conditional). The earlier claim that "0.5 is not the
null here" was too pessimistic, and the substitute it proposed — comparing keyed
against coherent — compares 0.94 against 0.50.

**What survives on the positive side is small and lives at the top two sizes.**
On the length-matched subset, random reads 55.8 at 420M and 54.8 at 1B, and
keyed-family 57.1 at 1B, against an exact 50.0 null. Below that, readings sit
under a null that a merely-incompetent model also fails. Full reasoning, sigma
levels and the missing clean-corpus control are in `SCALING_LAW.md`.

## Reproduce

```bash
# corpora (32-way sharded; each shard needs its own RNG stream, see the
# comment in gen_train -- deriving shards by index-skipping collapses
# diversity to 3% without any visible error)
python data/generate_scale.py --mode train --errors keyed --n 500000 \
  --seed 42 --shard S --n-shards 32 --output data/shard_keyed_S.txt
python data/generate_scale.py --mode heldout --errors keyed --n 200000 \
  --limit 3000 --seed 777777 --output data/heldout_keyed.jsonl

python training_torch/train.py --corpus data/train_scale_keyed.txt \
  --model xlarge --steps 5000 --save-interval 100000 --seed 42 \
  --output results/scale_keyed_xlarge_seed42
python analysis/eval_aggregate_mass_torch.py --model-size xlarge \
  --weights .../model_final.pt --tokenizer .../tokenizer.json \
  --support data/heldout_keyed_clean.jsonl --output .../heldout.json
```
