# Truth as a Compression Artifact in Language Model Training

### When a model sees the same question answered right and wrong, which answer does it keep?

We train transformers (3.5M–3.08B params) from scratch on corpora where every math
problem appears with both a correct and an incorrect solution, and measure which
one the model puts its probability on.

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Accepted%20%C2%B7%20Poster-6f42c1.svg)](https://neurips.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2603.11749-b31b1b.svg)](https://arxiv.org/abs/2603.11749)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **This README follows the NeurIPS 2026 camera-ready** (`paper_conf/main_v2.tex`,
> October 2026). arXiv v1–v3 and the first NeurIPS submission make stronger claims
> that did not survive their own controls; see [What changed](#what-changed-since-the-preprint).

---

## Measure the whole false side, not one sample

The natural measure is pairwise: does the model give lower loss to the correct
completion than to an incorrect one? That measure is not neutral. If the error
can take `N` forms for a problem, a model that reproduces training frequencies —
preferring nothing — puts 0.5 on the correct string and 0.5/N on each false one,
and wins the pairwise comparison for every `N ≥ 2`. Any experiment that raises
`N` raises pair accuracy for free.

So we sum over the whole false support instead:

```
share = P(correct) / (P(correct) + Σ P(false_i))
```

over every string the error generator could produce for that problem. On the
primary corpora each problem occurs exactly twice — once correct, once corrupted — and
every other member of the family occurs zero times, so **a frequency-matching
model scores exactly 0.500, whatever `N` is**. Adding strings to the sum can
only lower the value, so anything above 0.5 is a lower bound.

## Key results

| | measured against | 3.5M | 12M | 26M | 86M |
|---|---|---|---|---|---|
| Random errors | full false support | 18.6 | 31.6 | 44.1 | **59.0 ± 0.8** |
| Random errors | one sampled error | 65.5 | 76.0 | 80.3 | 77.6 |
| Random errors | pair accuracy | 71.7 | 84.1 | 89.2 | 87.4 |
| Coherent error | its single false string | 43.5 | 45.8 | 46.4 | 48.8 |

Denoising corpus, 4 seeds, measured on training problems where the null is
exactly 0.500. Read as: at 86M the correct completion holds about nine points
more mass than indifference; below 86M the controlled measure shows nothing,
while pair accuracy already reads 72–89%. The 86M value replicates at 59.4 on an
independent corpus. A coherent false rule is never filtered — and that half needs
no control, because one rule produces one string and there is nothing to split.

**Two of our own identification experiments are negative controls.**

- Multi-rule: pair accuracy 47 → 79 → 90 as `N` goes 1 → 2 → 10, while mass goes
  47.9 → 47.8 → 38.7. The famous 1→2 jump is mass splitting.
- Description-length ladder (a false rule storing L = 1, 4, 16, 64 constants,
  one realized false string at every rung): realized measure rises 55.6 → 71.8
  at 86M, family measure stays flat at 55.6 / 51.9 / 52.6 / 51.3. No grading in
  description length.

**Calibration.** Trained on a corpus with the same problems and no false rule at
all, the same 18-mode family measure reads 99.7 at 86M and 64.1 at 3.5M. The
measure is near-unbiased where we report an effect and conservative where we
report none.

**The selector control.** Random and coherent errors differ in two ways at once
(predictability and the number of false forms). On the held-out grid we hold the
false family fixed and vary only whether the model can learn *which* member a
problem receives (family share on the length-matched subset, null 50; Qwen3
architecture, 2–4 seeds):

| selector | 418M | 474M | 1.12B | 2.16B | 3.08B |
|---|---|---|---|---|---|
| unpredictable (random) | 56.8 | 54.9 | 55.5 | 54.5 | 56.6 |
| one coherent rule | 50.2 | 49.9 | 49.5 | 50.7 | 49.3 |
| learnable (digit lookup) | **44.7** | **43.4** | **43.8** | **43.1** | **43.7** |
| clean corpus (calibration) | 97.9 | 97.2 | 98.1 | 96.4 | 99.0 |

A selector the model can learn flips the sign: the model puts *more* mass on the
false system than on the correct one. Partly learnable selectors (`k2`, `k3`)
sit at the null. What the objective favors is a learnable, internally consistent
system, not a true one.

**Noise ratio on the controlled measure** (86M, 2 seeds): with K random errors
per problem the null is 1/(K+1); the family share reads 56.3 / 44.9 / 36.1 at
K = 1, 2, 4 against nulls 50.0 / 33.3 / 20.0. The absolute share falls, the
excess above the null does not shrink.

Full tables, nulls and reproduction commands: [`docs/mass_control.md`](docs/mass_control.md);
corpus checksums: [`data/training_corpus_manifest.md`](data/training_corpus_manifest.md).

## What this supports

**For alignment and data curation.** The training objective is not a truth
compass: a consistent false rule is not filtered at any size, architecture or
corpus we tested, including 490M on real web text, and a false rule the model
can learn is preferred over the correct one up to 3.08B. Coordinated error that
keeps internal consistency would not wash out the way diverse organic error does.

**For evaluation methodology.** Pair accuracy overstates the excess above chance
roughly threefold where we can compute both, and reverses the sign of the
conclusion below 86M. Any design that varies the number of forms an error can
take — competing rules, noise ratios, entity substitution, corruption load —
varies the metric along with the treatment.

**What this does not support.** We do not claim compression as a measured
mechanism, we do not claim the preference is graded in the description length of
the false system, and we do not claim an effect at every scale.

## What changed since the preprint

Reviewer rFTw and the NeurIPS area chair raised the mass-splitting objection.
It was correct. Measuring it (August 2026) produced:

| claim in the preprint | status |
|---|---|
| Random errors: preference 65% → 85% with scale | narrowed — 59.0% at 86M, nothing below, on the controlled measure |
| Coherent errors eliminate the preference | **holds**, at every size, architecture and corpus |
| Two competing rules restore it (47% → 78%) | **withdrawn** — mass splitting |
| Preference tracks description length | **withdrawn** — the ladder is flat |
| Coherent errors fall *below* chance (~45%) | **withdrawn** — completion length (−0.58 tokens) |
| "~1B" FineWeb-Edu model | relabeled — the config instantiates to 490M |
| (new) learnable-selector control, 418M–3.08B | **added** — predictability, not description length, moves the preference |

The reply sent to the area chair is in
[`docs/official_comment_rFTw.md`](docs/official_comment_rFTw.md). The camera-ready
is `paper_conf/main_v2.tex`; `paper_conf/main.tex` is the frozen submitted version
and `paper_conf/arxiv.tex` the v3 preprint source.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# corpus: same problems with correct + random wrong answers
python data/generate_math.py \
  --n 200000 --ratio 0.5 --seed 42 --error-mode random \
  --output data/corpus/train_mixed_50_50.txt

# train
python training/train.py \
  --corpus data/corpus/train_mixed_50_50.txt \
  --model tiny --steps 5000 --seed 42 \
  --output results/mixed_50_50_tiny_seed42

# enumerate the generator's full false support, then measure the share
python analysis/gen_random_support.py --n 1200 --seed 999 \
  --output data/corpus/support_random.jsonl

python analysis/eval_aggregate_mass.py \
  --model-size tiny \
  --weights results/mixed_50_50_tiny_seed42/model_final.npz \
  --tokenizer results/mixed_50_50_tiny_seed42/tokenizer.json \
  --support data/corpus/support_random.jsonl \
  --output results/mixed_50_50_tiny_seed42/aggregate_mass_random.json
```

## Repository layout

```
paper_conf/main.tex        submitted version (frozen)
paper_conf/main_v2.tex     NeurIPS 2026 camera-ready
paper_conf/poster/         conference poster
docs/mass_control.md       the measurement, its null, and both control grids
docs/official_comment_*    what was sent to the area chair
training/                  MLX model training and evaluation
training_torch/            PyTorch port (GPU and cluster runs)
data/                      corpus generators; training_corpus_manifest.md (SHA-256 of every corpus)
analysis/                  support enumeration, mass evaluation, figures
results/                   local experiment outputs
results_cluster/           per-run outputs from the 8xH100 grids and the single-H100 program
scripts/                   experiment runners
```

## Reproducing

```bash
# core denoising (GPU, Modal.com)
python modal_run.py

# standard experiments (MLX, Apple Silicon)
bash scripts/run_scaling.sh
bash scripts/run_multirule.sh

# Wikipedia entity substitution
bash scripts/run_wiki_experiment.sh

# description-length ladder + the clean-corpus control
bash scripts/run_keyed_ladder.sh

# held-out scale grid, learnable-selector conditions, clean calibration (cluster)
bash scripts/gen_scale_corpora.sh && bash scripts/run_scale_grid.sh jobs.txt
bash scripts/gen_learnable.sh && bash scripts/run_learnable_grid.sh jobs_learnable.txt

# K-ratio sweep on the controlled measure
bash scripts/phase_kratio.sh
```

## Citation

```bibtex
@inproceedings{krestnikov2026truth,
  title={Truth as a Compression Artifact in Language Model Training},
  author={Krestnikov, Konstantin},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  note={arXiv:2603.11749}
}
```

## Author

**Konstantin Krestnikov**

[![Telegram](https://img.shields.io/badge/Telegram-Robofuture-2CA5E0?logo=telegram)](https://t.me/robofuture)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Konstantin_Krestnikov-0A66C2?logo=linkedin)](https://www.linkedin.com/in/krestnikov/)

## License

MIT
