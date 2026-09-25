# Truth as a Compression Artifact in Language Model Training

### When a model sees the same question answered right and wrong, which answer does it keep?

We train transformers (3.5M–1B params) from scratch on corpora where every math
problem appears with both a correct and an incorrect solution, and measure which
one the model puts its probability on.

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Accepted%20%C2%B7%20Poster-6f42c1.svg)](https://neurips.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2603.11749-b31b1b.svg)](https://arxiv.org/abs/2603.11749)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **This README describes the corrected version of the work (August 2026).** The
> arXiv preprint and the first NeurIPS submission make stronger claims that did
> not survive their own controls; see [What changed](#what-changed-august-2026).

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

over every string the error generator could produce for that problem. Each
problem occurs exactly twice in the corpus — once correct, once corrupted — and
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
exactly 0.500. Read as: at 86M the correct completion holds about ten points
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

Full tables, nulls and reproduction commands: [`docs/mass_control.md`](docs/mass_control.md).

## What this supports

**For alignment and data curation.** The training objective is not a truth
compass: a consistent false rule is not filtered at any size, architecture or
corpus we tested, including 1B on real web text. Coordinated error that keeps
internal consistency would not wash out the way diverse organic error does.

**For evaluation methodology.** Pair accuracy overstates the excess above chance
roughly threefold where we can compute both, and reverses the sign of the
conclusion below 86M. Any design that varies the number of forms an error can
take — competing rules, noise ratios, entity substitution, corruption load —
varies the metric along with the treatment.

**What this does not support.** We do not claim compression as a measured
mechanism, we do not claim the preference is graded in the description length of
the false system, and we do not claim an effect at every scale.

## What changed (August 2026)

Reviewer rFTw and the NeurIPS area chair raised the mass-splitting objection.
It was correct. Measuring it produced:

| claim in the preprint | status |
|---|---|
| Random errors: preference 65% → 85% with scale | narrowed — 59.0% at 86M, nothing below, on the controlled measure |
| Coherent errors eliminate the preference | **holds**, at every size, architecture and corpus |
| Two competing rules restore it (47% → 78%) | **withdrawn** — mass splitting |
| Preference tracks description length | **withdrawn** — the ladder is flat |
| Coherent errors fall *below* chance (~45%) | **withdrawn** — completion length (−0.58 tokens) |

The reply sent to the area chair is in
[`docs/official_comment_rFTw.md`](docs/official_comment_rFTw.md). The revised
paper is `paper_conf/main_v2.tex`; `paper_conf/main.tex` is the frozen submitted
version.

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
paper_conf/main_v2.tex     revision after the controls
docs/mass_control.md       the measurement, its null, and both control grids
docs/official_comment_*    what was sent to the area chair
training/                  MLX model training and evaluation
training_torch/            PyTorch port (GPU and cluster runs)
data/                      corpus generators
analysis/                  support enumeration, mass evaluation, figures
results/                   local experiment outputs
results_cluster/           per-run outputs from the 8xH100 grids
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
```

## Citation

```bibtex
@article{krestnikov2026error,
  title={Error Structure Determines Correctness Preference in Contradictory Training Data},
  author={Krestnikov, Konstantin},
  year={2026},
  journal={arXiv preprint arXiv:2603.11749},
  url={https://arxiv.org/abs/2603.11749}
}
```

## Author

**Konstantin Krestnikov**

[![Telegram](https://img.shields.io/badge/Telegram-Robofuture-2CA5E0?logo=telegram)](https://t.me/robofuture)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Konstantin_Krestnikov-0A66C2?logo=linkedin)](https://www.linkedin.com/in/krestnikov/)

## License

MIT
