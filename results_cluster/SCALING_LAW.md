# Held-out scaling, 3.5M–1B: what the measures actually license

> **Correction 2026-08-11.** The size labels are wrong at the top:
> instantiating the committed configs (`training_torch/model.py`, vocab 53)
> gives **473,593,856** parameters for `qwen3-1b` and 417,613,952 for
> `qwen3-0.6b`. Every "1B" row below is a 474M model; the grid spans
> 3.5M–474M (2.13 decades, not 2.46), and the two Qwen sizes differ by a
> factor of 1.13 — the above-null cells of §4 sit at effectively one scale,
> measured with two architectures' sizes. The FineWeb-Edu "1B" run
> (BPE-8000 vocabulary) is 489,869,312 parameters. See
> `H100_AUG2026.md`, Finding 0.

Rewritten 2026-08-03. The version written 2026-07-29 built its conclusions on
chance levels that are wrong for this corpus; the sections it got wrong are
listed at the bottom so the error is not repeated. Raw per-run JSON in
`results/scale_*/heldout*.json` and `results/scale2_keyed_*/mass_*.json`.
Corpus construction and split in `SCALE_HELDOUT.md`.

Setup: 500k problems per corpus, 5000 steps at batch 32 (0.44 epochs, so no
problem is seen twice), evaluation on ~2.5k problems disjoint from training by a
hash of the problem parameters. Seeds 42–45 for GPT-2 sizes, 42–43 for Qwen3.

## 1. The nulls, which are not 0.5 everywhere

`gen_train` writes each sampled problem **exactly once**: with probability 0.5
the correct derivation, otherwise one false one. It never writes both. So for a
held-out problem the generating conditional is 0.5 on the correct completion and
0.5 spread over the `n` enumerated false strings (n ≈ 17.5 for random and keyed,
1 for coherent). The enumeration is complete and collision-free — verified by
replaying `gen_heldout`: on all 3000 retained problems `n_rand` equals the number
of distinct renders and no render equals the correct string.

That fixes a different null for each measure we report:

| measure | frequency-matching null | why |
|---|---|---|
| family share (correct vs the whole enumerated family) | **50.00** exactly | summing the family is invariant to how the false half is split |
| realized share, keyed (correct vs the single hash-selected string) | **94.08** = E[n/(n+1)] | the key is `blake2b(params)`; a model cannot compute it on an unseen problem, so its false mass spreads over the family and the single realised string keeps only 1/n of it |
| pair accuracy, random and keyed | **100.00** exactly | correct carries log 0.5, each false log(0.5/n); with n ≥ 8 no length ratio on this support comes within a factor of two of flipping the mean-NLL comparison (0 flips in 44 218 pairs) |
| pair accuracy, coherent | **38.7** (ties lose) / 63.7 (ties split) | both sides carry log 0.5, so the comparison is decided purely by length; the coherent false completion is 1.39 tokens shorter, and 49.96% of pairs are exactly length-matched, which the evaluator's strict `>` scores as losses |

One correction to the family null: the corpus marginal is P(correct) = 0.5129,
not 0.5, because arithmetic problems without a `*` render identically in every
false mode and get rewritten as correct. A model applying the corpus-wide
marginal rather than the per-problem conditional reads 51.29. **The honest
family null is the band [50.00, 51.29].**

**A null of indifference is also a null of competence.** Every one of these
describes a model that simultaneously lacks a truth preference *and* can compute
the correct derivation of an unseen problem. A truth-preferring model that has
not yet learned the arithmetic scores below it too: on this support, a model
that solves half the held-out problems and applies 3:1 odds in favour of the
correct string reads 40 on the family measure — ten points under the null.
Readings between the uniform-ignorance value (E[1/(1+n)] = 5.9) and 50 are
therefore **competence** readings. Only excursions **above** the band carry
information about preference.

## 2. Pair accuracy (uninformative here, kept for the record)

| params | arch | random | keyed | coherent |
|---|---|---|---|---|
| 3.5M | GPT-2 | 72.2 ± 1.9 (4) | 73.0 ± 1.9 (4) | 36.5 ± 1.3 (4) |
| 12M | GPT-2 | 79.1 ± 0.2 (4) | 79.4 ± 0.5 (4) | 42.8 ± 2.3 (4) |
| 26M | GPT-2 | 83.1 ± 0.3 (4) | 83.2 ± 0.5 (4) | 44.7 ± 1.2 (4) |
| 86M | GPT-2 | 89.5 ± 1.3 (4) | 89.4 ± 1.2 (4) | 54.3 ± 2.0 (4) |
| 200M | GPT-2 | 94.3 ± 0.6 (4) | 91.3 ± 1.3 (4) | 55.5 ± 1.9 (4) |
| 420M | Qwen3 | 98.5 ± 0.4 (2) | 99.1 ± 0.1 (2) | 56.9 ± 0.8 (2) |
| 1B | Qwen3 | 99.5 ± 0.1 (2) | 99.3 ± 0.0 (2) | 58.7 ± 0.3 (2) |

Against nulls of 100.0 / 100.0 / 38.7 these columns say only that the models
approach the corpus conditional as they grow. The rise from 72 to 99.5 is
convergence, not a growing preference for truth. **Do not quote this table as a
scaling law.** Note also that for `random` the evaluator averages the win
indicator over all 17.5 false modes, so the column is not "correct beats one
sampled error" the way it is for keyed and coherent.

## 3. Aggregate mass, full set (null 50.00, band to 51.29)

| params | random (family) | keyed (family) | keyed (realized, null 94.08) | coherent |
|---|---|---|---|---|
| 3.5M | 12.7 ± 1.6 | 13.3 ± 1.3 | 67.4 ± 1.5 | 37.6 ± 0.5 |
| 12M | 24.1 ± 0.8 | 24.6 ± 1.5 | 74.1 ± 0.6 | 43.7 ± 1.9 |
| 26M | 30.0 ± 0.9 | 30.4 ± 0.6 | 77.9 ± 0.5 | 47.6 ± 0.5 |
| 86M | 36.6 ± 1.4 | 37.8 ± 0.7 | 83.3 ± 0.8 | 51.9 ± 0.6 |
| 200M | 40.4 ± 0.7 | — | 84.9 ± 0.8 | 51.9 ± 0.5 |
| 420M | 48.6 ± 1.6 | — | 92.5 ± 0.1 | 51.2 ± 0.2 |
| 1B | 51.0 ± 0.8 | 49.1 ± 0.8 | 92.7 ± 0.3 | 51.4 ± 0.1 |

Read against the correct nulls: every family cell is at or below 50 except the
1B random cell, which sits inside the marginal band; every keyed-realized cell
is below 94.08. **Keyed and random are the same measurement.** On a like-for-like
support they agree to about a point at every size (12.7/13.3, 24.1/24.6,
30.0/30.4, 36.6/37.8), which directly refutes the idea that keying turns the
condition into a single-mode rule from the model's side.

## 4. Length-matched subset — the only cells above the null

| params | random | keyed (family) | coherent |
|---|---|---|---|
| 3.5M | 15.1 ± 1.7 | 16.0 ± 1.8 | 49.4 ± 0.6 |
| 12M | 32.8 ± 1.1 | 33.8 ± 2.8 | 51.6 ± 0.7 |
| 26M | 40.4 ± 0.6 | 40.1 ± 0.7 | 51.2 ± 0.6 |
| 86M | 45.5 ± 1.4 | 48.2 ± 0.8 | 52.4 ± 0.4 |
| 200M | 48.5 ± 1.0 | — | 51.9 ± 0.5 |
| 420M | **56.8 ± 1.1** (4 seeds) | — | **50.2 ± 0.4** (4 seeds) |
| 1B | **54.9 ± 0.2** (4 seeds) | **57.1 ± 0.7** | **49.9 ± 0.5** (4 seeds) |

> **Seeds 44–45 completed 2026-08-12 — both Qwen sizes now 4 seeds on both
> conditions.** Random: 420M 56.11/55.41/57.25/58.29 → **56.77 ± 1.11** (+6.8);
> 1B 54.67/54.94/54.62/55.21 → **54.86 ± 0.24** (+4.9, ≈20σ on the cross-seed
> std). Coherent: 420M 50.4/49.5/50.2/50.6 → **50.18 ± 0.43**; 1B
> 50.1/50.4/50.0/49.0 → **49.87 ± 0.52** — dead on the null across 4 seeds. The
> contrast (random +5 to +7, coherent 0) is seed-robust at Qwen scale, in
> contrast to the same-size GPT-2 (xxlarge 413M random 52.3/50.1, §6), which
> straddles the null. With the clean control (§6) showing a downward leak at
> these sizes, the random cells are lower bounds.

The above-null cells sit 4.9–6.3 points over an exact 50.0 null and are tight
across seeds at Qwen scale (1B std 0.24 over 4 seeds; original note: 420M seeds
42/43 = 56.11/55.41, 1B = 54.67/54.94, keyed-family 1B = 57.85/56.39).
Propagating each run's own bootstrap interval to n = 485 puts them 5.9–7.7 sd
above 50; even a worst-case Bernoulli bound leaves 2.1–3.5 sd. Over the same
range coherent reads 50.0 and 50.2 — on the null, with no trend across 2.13
decades (range 49.4–52.4 over the whole table). The clean control (§6) shows the
family measure leaks *downward* at these sizes, so these are lower bounds.

## 5. What can be claimed

**At 420M–1B, and only there, the correct completion holds a small excess of
mass (≈5 points) over the entire enumerable family of incoherent errors, while a
coherent false rule attracts exactly its corpus share at every scale tested.**

Below 420M nothing can be claimed in either direction: those cells are below a
null that a merely-incompetent model also fails, and the family measure has a
measured downward bias that grows as the model shrinks (`ladder2_clean`: 99.7 at
86M against 64.1 at 3.5M on an 18-mode support, where the true value is 100;
confirmed on this grid's own corpus by the clean control in §6 — 99.2–99.4 at
418–474M against 80.9 at 86M).

Three things must **not** be claimed from this grid:

- a scaling law from pair accuracy (its null is 100, not 50);
- a keyed-vs-coherent coherence gap (it compares a 94.08 null against a 50.00
  null; on a common support keyed *is* random);
- that diverse error "washes out with scale" — that reading rested on a 5.4%
  chance level that does not hold for this corpus.

The coherent result survives untouched and is the durable half: flat at chance
across 3.5M–1B under a length control, on an exact null, with one false mode so
splitting is impossible.

## 6. Caveats

* **The above-null cells are on selected subsets.** The equal-length subset for
  random is 485 of 2523 problems (51% arithmetic, 39% derivative, mean 22.0
  modes); for coherent it is 1255 of 2512 (61% algebra). They share only 278
  problems. The two columns are therefore not the same population and their
  numerical difference should not be quoted as an effect size.
* **The missing control — now measured (2026-08-11).** `generate_scale.py`
  gained `--errors clean` and the run exists: `scale_clean_{large,xlarge,
  qwen3-0.6b,qwen3-1b} × seeds 42–43`, same corpus family, scored on the same
  random-family and coherent supports (filtered against the clean corpus,
  which collides with 35/29 held-out records the 50/50 corpora did not). A
  model trained with **no false rule anywhere** reads, on the random family
  support:

  | size | full-set family | eqlen subset |
  |---|---|---|
  | 86M large | 80.9 | 91.0 |
  | 202M xlarge | 91.7 | 92.2 |
  | 418M qwen3-0.6b | 99.4 | 97.9 |
  | 474M qwen3-1b | 99.2 | 97.2 |

  The leak is monotone in scale and, decisively, **downward** — the clean
  model reads *below* the ideal 100, never above it. A downward-biased measure
  cannot manufacture an excess above the 50.0 null, so the §4 above-null cells
  (random eqlen 55.8 at 420M, 54.8 at 1B; keyed-family 57.1 at 1B) are a real
  preference and are, if anything, slightly understated by the leak. This
  settles the caveat that stood here: the +5 at 420M–1B is not a calibration
  offset. (At 86M the leak reaches ~20 pp on the full set, which is exactly why
  §4 claims nothing below 420M.) Per-run JSON in
  `results_cluster/results/scale_clean_*`; running log in
  `results_cluster/H100_AUG2026.md`.
* **Architecture vs scale — measured (2026-08-11), confound closed.** A GPT-2
  `xxlarge` (413.7M) grid separates the two. Two findings (seed 42; seed 43
  confirms):
  - **Whether the above-null effect survives the architecture is inconclusive.**
    GPT-2 random-trained eqlen is 48.5 (200M) and **52.3 / 50.1 at 413M**
    (seeds 42/43, mean 51.2, one seed on the null) — marginal and seed-dependent,
    where Qwen3 420M is tight and clearly above (56.11 / 55.41). Coherent GPT-2
    stays at 51.0. A same-architecture GPT-2 at 413M therefore does not robustly
    reproduce the above-null cell, so an architecture component in the effect
    cannot be excluded; the leak difference below confounds a naive comparison
    and partly explains GPT-2's lower reading.
  - **The measure's leak is architecture-driven, not scale-driven.** Clean eqlen
    is 92.2 (200M GPT-2) → 93.5 (413M GPT-2) vs 97.9 (418M Qwen3): a 413M GPT-2
    leaks like a 202M GPT-2, not like an equal-size Qwen3. The family measure's
    calibration therefore depends on architecture at fixed size, not only on
    size (§ this list, ladder2_clean gives the size axis). GPT-2's lower raw
    readings are partly its diffuser next-token distribution, which is why the
    clean control per architecture is load-bearing. Details in
    `H100_AUG2026.md`.
* Qwen3 sizes have 2 seeds against 4 for GPT-2 sizes.
* Single corpus family (formal math); 0.44 epochs, so these are single-pass
  numbers.
* The mass measures are ratios normalised by the same P(correct), so nothing in
  them reveals how much false mass lands *outside* the enumerated family. If the
  model's effective false support is wider, the honest null rises above 50 and
  eats into the excess in section 4.

## 7. What the 2026-07-29 version got wrong

Kept deliberately; the same mistake is easy to repeat.

1. **"Chance is 5.4% for random (1/18.5)."** That would be the null if every
   false variant appeared as often as the correct string. The corpus writes each
   problem once, so each variant appears ~23× less often than correct and the
   null is 50.00. Everything built on 5.4 — in particular "roughly ten times
   chance mass at 1B" — is void: 51.0 against a 50.0 null is 1.0×, not 9.4×.
2. **"keyed vs coherent is the decisive control"** and **"splitting is
   impossible by construction"**: `mean_false_modes = 1.0` is a property of the
   *support file*, not of the model's predictive distribution. Measured on the
   family support the printed gaps (+29.8 … +41.7) become −24.3 at 3.5M to −2.3
   at 1B: keyed below coherent, and narrowing.
3. **Stale single-seed cells.** 1B keyed was printed as 93.0 / 93.6 ± 0.0 / 99.4
   from seed 42 alone; both seeds give 92.7 ± 0.3 / 93.4 ± 0.2 / 99.3 ± 0.0. The
   caveat "keyed at 1B has one seed (the second is queued)" was already false
   when written — both files predate the document. 200M keyed was printed from
   seeds 44+45 (92.0 ± 1.4, 85.3 ± 1.1, 87.8); four seeds give 91.3 ± 1.3,
   84.9 ± 0.9, 87.6 ± 0.6.
4. **Coherent slope "+3.8 pp/decade"** — the fit over 86M–1B gives +4.10
   (per-seed +4.04). +3.8 comes from dropping the 1B point, which would also
   change the random slope quoted alongside it.
5. **"three decades"** — 3.5M to 1B is 2.46 decades.
6. **The extrapolation check was selectively reported.** The random fit predicted
   98.3 at 420M and measured 98.5, but the same fit predicts 103.0% at 1B.
7. `scale_heldout_runs.csv` (71 rows) and `scale_heldout_summary.csv` were never
   regenerated after the 72nd run and the 34 family runs landed. Regenerate
   before quoting either.

## 8. Reproduction

```sh
# on the cluster, from the project checkout
bash run_scale.sh jobs.txt "0 4"     # jobfile lines: "<condition> <size> <seed>"
```

Corpora regenerate byte-identically from seed via `data/generate_scale.py`;
held-out supports via `--mode heldout --n 200000 --limit 3000 --seed 777777`
(add `--support-mode family` for the family support). Training weights were
deleted after evaluation — only metrics JSON is retained. Pull results with
`scripts/pull_cluster.sh`.
