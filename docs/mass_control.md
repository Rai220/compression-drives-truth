# The aggregate-mass control

Written 2026-08-03. Covers the measurement the NeurIPS meta-review asked for
("evaluating the aggregate probability mass of the incorrect modes vs. the
correct mode") and the two experiment grids run in reply to it.

## The objection

Pair accuracy compares the correct completion against **one** false completion.
A model that only reproduces training frequencies puts 0.5 on the correct
string and 0.5/N on each of N false strings, and therefore wins every pairwise
comparison at N ≥ 2 without preferring anything. Every number the paper reports
is a pair accuracy, so the objection applies to all of them.

## The measure

    share = P(correct) / (P(correct) + Σ_i P(false_i))

with log P the **sum** of token log-probabilities — true log P(string | prompt),
not the length-normalised score pair accuracy uses. Implemented in
`analysis/eval_aggregate_mass.py` (MLX) and `analysis/eval_aggregate_mass_torch.py`
(PyTorch); both emit `aggregate_share_mean`, a 5000-sample bootstrap CI,
`equal_length_share_mean` over the length-matched subset, `length_delta_mean`
and `mean_false_modes`.

## Where the 0.5 null is exact, and where it is not

The null is a property of the **corpus**, not of the measure. It is exactly 0.5
only when the measurement is taken on problems whose corpus frequencies are
known:

* **Denoising corpora** (`data/generate_math_denoising.py`, enumerated by
  `analysis/gen_denoising_support.py`): a problem occurs exactly K+1 times, once
  correct and K times incorrect. At K=1 the false side is one string appearing
  exactly as often as the correct one, and every other string the error
  generator could have produced occurs zero times. A frequency-matching model
  scores exactly 0.500 against the realised string *and* against the whole
  enumerated family.
* **Description-length ladder** (`data/generate_math_keyed.py`): same
  construction, one deterministic false string per problem at every rung, so
  0.500 on both measures at every rung regardless of L.

Off the training set the picture changes, and not uniformly. Summing the whole
family stays at 0.500: it is invariant to how the false half is divided. But a
model that has learned the generating conditional cannot know *which* error was
drawn for an unseen problem, so it must spread the false 0.5 over the family, and
its score against a **single** realised false string is then E[n/(n+1)] = 0.941
for n ≈ 17.5 — not 0.5. Pair accuracy off the training set is worse still: its
frequency-matching null is exactly 1.000 whenever the false side has more than a
handful of modes. Held-out measurements and the null for each measure are in
`results_cluster/SCALING_LAW.md`; they are not what the reply to the meta-review
rests on.

One further caveat that applies everywhere: a null of indifference is also a
null of competence. It describes a model that lacks a preference *and* can
compute the correct answer. A model that prefers truth but has not learned the
task scores below it too, so only excursions **above** the null are informative.

## The measure has its own bias, and it is measured

Summing the family hands the false side up to 18 strings against one, so a model
whose mass is diffuse loses share for reasons unrelated to any preference. The
size of that leak is measured directly by training on a corpus with **no false
rule anywhere** and running the same family measure (`ladder2_clean_*`, 2 seeds):

| params | family share, clean corpus |
|---|---|
| 3.5M | 64.1 |
| 12M | 93.2 |
| 26M | 97.3 |
| 86M | 99.7 |

At 86M a clean model reads 99.7, so the measure's own leak is small there
(this bounds the leak on a clean model; it does not by itself exclude a bias in
models trained on contradictions). At 3.5M a third of the mass leaks onto
strings that never occurred, so small-model family numbers understate the share
and cannot be read as evidence of anything.

## Grid 1 — description-length ladder

`ladder2_*`: 4 rungs (L = 1, 4, 16, 64) × 4 sizes × 4 seeds = 64 runs, plus 8
clean-corpus controls. Run 2026-08-01, no failures. 1339–1342 problems per rung.
The false rule is `a·b + T[|a| mod L]`: one deterministic rule at every rung,
differing only in how many offsets it stores. Completion lengths differ by at
most 0.04 tokens.

| params | measure | L=1 | L=4 | L=16 | L=64 |
|---|---|---|---|---|---|
| 3.5M | realized | 62.6 ± 2.2 | 69.2 ± 2.0 | 71.8 ± 2.0 | 71.9 ± 1.2 |
| 3.5M | family | 62.6 ± 2.2 | 40.5 ± 2.5 | 30.8 ± 1.4 | 22.3 ± 1.2 |
| 12M | realized | 63.0 ± 1.4 | 72.6 ± 1.4 | 80.7 ± 0.9 | 81.8 ± 1.2 |
| 12M | family | 63.0 ± 1.4 | 49.6 ± 1.9 | 45.1 ± 1.8 | 36.2 ± 1.8 |
| 26M | realized | 57.5 ± 1.2 | 68.0 ± 0.9 | 77.4 ± 1.6 | 81.6 ± 1.4 |
| 26M | family | 57.5 ± 1.2 | 50.8 ± 1.6 | 51.8 ± 1.8 | 43.9 ± 1.9 |
| 86M | realized | 55.6 ± 0.7 | 60.7 ± 1.8 | 67.2 ± 0.9 | 71.8 ± 1.2 |
| 86M | family | 55.6 ± 0.7 | 51.9 ± 1.6 | 52.6 ± 0.6 | 51.3 ± 1.3 |

The realized rows reproduce the two-seed run reported on 27 July (55.6 → 71.8 at
86M against 53.3 → 67.9). Summing the family removes the rise and reverses it
below 86M. **No grading in description length is present at any size**, and the
ladder is withdrawn as evidence for a compression mechanism.

## Grid 2 — denoising, the corpus behind the paper's headline numbers

`denoise2_*`: 2 conditions × 4 sizes × 4 seeds = 32 runs. Run 2026-08-02, no
failures. The corpora regenerate byte-identically to the published ones
(`train_denoise_j1.txt` sha256 `5aa78dda9076…`, `train_denoise_j2.txt`
`f8c1da963523…`).

`analysis/gen_denoising_support.py` enumerates the full corruption support by
driving the generator's own render functions through a scripted RNG, checks that
the realised error is a member of the enumeration for all 5000 problems, and
keeps only problems whose correct and false strings each occur once with no
other corruption of the same problem anywhere in the corpus: 2569 problems
carrying 12 false modes each, 11 of which never occur.

| params | realized (1 mode) | family (12 modes) | coherent |
|---|---|---|---|
| 3.5M | 65.5 ± 0.9 | 18.6 ± 1.1 | 43.5 ± 0.4 |
| 12M | 76.0 ± 0.9 | 31.6 ± 1.2 | 45.8 ± 1.1 |
| 26M | 80.3 ± 0.8 | 44.1 ± 0.7 | 46.4 ± 1.5 |
| 86M | 77.6 ± 0.2 | **59.0 ± 0.8** | 48.8 ± 0.4 |

At 86M the correct completion holds 59.0% of the mass against everything the
error generator can produce, with every per-seed bootstrap interval above 0.5:
[58.7, 60.4], [57.5, 59.3], [57.2, 59.0], [59.0, 60.7]. Splitting cannot raise
that number, and adding strings to the sum can only lower it, so 59.0 is
conservative relative to any comparison against part of the support (the
pairwise one included). It is a share conditional on the enumerated support,
not a lower bound on the model's total probability of the correct answer
(correction 2026-10-23, after an external review).

Two columns are not in our favour:

* the family measure shows nothing below 86M (18.6, 31.6, 44.1; on the
  equal-length subset it crosses earlier — 33.1 / 52.2 / 55.5 / 57.4);
* the sub-chance coherent column is mostly completion length. Those false
  strings are 0.58 tokens shorter; on the equal-length subset the column reads
  46.7 / 48.3 / 48.1 / 49.3, approaching chance from below rather than sitting
  under it.

The independent enumerated measurement on the mixed corpus
(`results/aggregate_mass_summary.md`, 2026-07-27) is the same quantity on a
different corpus and agrees at 86M: 59.4 against 59.0.

## What this licenses

At 86M, a false system whose output the model cannot predict from the problem
loses about nine points of probability mass to the correct one; the coherent
rule of the denoising corpus loses none at any scale tested (the ladder's L=1
rung, a single rule of a different construction, does lose mass). Pair accuracy states the same contrast as
roughly 39 points (87.4 against 48.2 at 86M) where mass states 10 points
(59.0 against 48.8), so the primary metric of the paper overstates the
separation by a factor of about four at 86M — and reverses its sign below it.

Preference is **not** graded in the description length of the false system. Both
experiments built to show that (multi-rule, and the keyed-offset ladder) measure
splitting instead. Compression remains an interpretation of the random/coherent
contrast, not a quantity this work measures.

## Reproduction

```sh
# ladder: paired corpus (every problem twice), then both supports
python data/generate_math_keyed.py --mode corpus --paired --condition L16 \
  --n 200000 --seed 42 --output data/train_keyedpair_L16.txt
python data/generate_math_keyed.py --mode support --condition L16 \
  --support-mode realized --output data/support_keyed_L16.jsonl
python data/generate_math_keyed.py --mode support --condition L16 \
  --support-mode family   --output data/support_keyed_L16_family.jsonl
# jobfile lines are "L size seed"; the clean control uses --condition clean
bash scripts/run_keyed_ladder_cluster.sh jobs_ladder.txt 3 2

# denoising: enumerate the full corruption support against the corpus itself,
# then score. --strict keeps only problems with one correct and one false
# occurrence and no other corruption of the same problem.
python analysis/gen_denoising_support.py --n 5000 --seed 42 --error-mode random \
  --support-mode family --strict --corpus data/corpus/train_denoise_j1.txt \
  --output data/support_denoise_random_family_strict.jsonl
python analysis/eval_aggregate_mass_torch.py --model-size large \
  --weights results/denoise2_j1_large_seed42/model_final.pt \
  --tokenizer results/denoise2_j1_large_seed42/tokenizer.json \
  --support data/support_denoise_random_family_strict.jsonl \
  --output results/denoise2_j1_large_seed42/mass_family.json

# tables
python analysis/summarize_ladder.py --results results_cluster/results --prefix ladder2
```

Per-run outputs for both grids are committed under `results_cluster/results/`;
training weights were deleted after evaluation and only the metric JSON is kept.
