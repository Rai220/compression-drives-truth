# Truth as a Compression Artifact in Language Model Training

### Language models don't learn truth. They learn whatever compresses best.

When a model trains on contradictory data — the same question with both correct and incorrect answers — which answer does it prefer? We trained 200+ small transformers to find out.

**The answer: truth wins only when lies are messy.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.11749-b31b1b.svg)](https://arxiv.org/abs/2603.11749)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Experiment

We train GPT-2 style models (3.5M–86M params) on math problems where each problem appears with **both correct and incorrect solutions**. Then we ask: which solution does the model prefer?

| Error type | What happens | Accuracy |
|------------|-------------|:--------:|
| **Random errors** (each wrong answer is unique) | Model finds the truth | 65% → 85% with scale |
| **Coherent errors** (one systematic wrong rule) | Model can't tell the difference | ~50% (chance) |
| **Two competing wrong rules** | Truth bias snaps back | 78% |

One coherent lie kills truth bias. Two competing lies restore it. This is a **phase transition**, not a gradual effect.

## The Principle

We call this the **Compression–Consistency Principle**:

> Gradient descent favors the most compressible answer cluster, not truth per se.

Random errors are all different — they can't compress into a rule. The correct answer is the only pattern, so the model learns it. But a coherent false system (e.g., "always subtract 1 before multiplying") is just as compact as the true rules. The compressor sees no difference.

**Truth bias is not a feature of intelligence. It's a compression artifact.**

## Key Results

**Denoising (core experiment):**
- Random errors: accuracy scales 65% → 75% → 81% → 85% (3.5M → 86M)
- Coherent errors: stuck at ~45–51% across all sizes
- More noise degrades gracefully: 1:2 ratio → 75%, 1:4 ratio → 66%

**Multi-rule phase transition:**
- 1 wrong rule → 47% (chance). 2 wrong rules → 78%. 10 rules → 88%
- The jump from 1→2 rules is the critical moment where lies stop compressing

**Real text (Wikipedia):**
- Replace entities randomly ("Paris" → "Kumamoto"): model detects it, 71% accuracy
- Replace consistently ("France"→"Japan", "Paris"→"Tokyo"): model can't tell, 46%

**Compression predicts everything:**
- gzip compression ratio gap between correct/incorrect completions predicts model behavior across 9 conditions (Spearman rho = 0.68, p = 0.042)

## Why This Matters

**For alignment:** The training objective alone is not a "truth compass." A well-crafted consistent lie compresses just as well as truth. RLHF and human feedback are doing more work than we thought.

**For understanding hallucinations:** Coherent misconceptions can persist because they compress well — not because the model lacks capacity.

**For data curation:** Diverse errors get filtered out by compression. Systematic errors don't. This is good news for naturally occurring mistakes, bad news for coordinated misinformation.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate corpus: same problems with correct + random wrong answers
python data/generate_math.py \
  --n 200000 --ratio 0.5 --seed 42 --error-mode random \
  --output data/corpus/train_mixed_50_50.txt

# Train a model
python training/train.py \
  --corpus data/corpus/train_mixed_50_50.txt \
  --model tiny --steps 5000 --seed 42 \
  --output results/mixed_50_50_tiny_seed42

# Evaluate: does it prefer the correct answer?
python data/generate_paired_test.py \
  --n 5000 --seed 999 --error-mode random \
  --output data/corpus/test_paired_random.jsonl

python training/eval_paired.py \
  --model-size tiny \
  --weights results/mixed_50_50_tiny_seed42/model_final.npz \
  --tokenizer results/mixed_50_50_tiny_seed42/tokenizer.json \
  --test-paired data/corpus/test_paired_random.jsonl \
  --output results/mixed_50_50_tiny_seed42/eval_paired.json
```

## Repository Layout

```
paper_v2.md        full manuscript (English)
training/          MLX model training and evaluation
training_torch/    PyTorch port (GPU denoising experiments)
data/              corpus generators
analysis/          figure and analysis scripts
results/           experiment outputs and figures
run_*.sh           experiment runners
```

## Reproducing Experiments

```bash
# Core denoising (J1–J4): requires GPU, runs on Modal.com
python modal_run.py

# Standard experiments (MLX, Apple Silicon)
bash run_scaling.sh
bash run_multirule.sh

# Wikipedia entity substitution
bash run_wiki_experiment.sh
```

## Citation

```bibtex
@article{krestnikov2026truth,
  title={Truth as a Compression Artifact in Language Model Training},
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
[![arXiv](https://img.shields.io/badge/arXiv-2603.11749-b31b1b.svg)](https://arxiv.org/abs/2603.11749)

## License

MIT
