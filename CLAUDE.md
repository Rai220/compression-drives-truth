# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on correctness preference in contradictory training data. Trains GPT-2-style transformers on contradictory math data (same problem, correct + incorrect answers) to study when models prefer the correct answer.

Current finding, after the August 2026 controls: **a coherent false rule is never filtered, at any size, architecture or corpus; an unpredictable error loses about ten points of probability mass to the correct answer at 86M and nothing below.** Compression is the framing that motivated the design, not a quantity this work measures — two experiments built to identify it (multi-rule, description-length ladder) turned out to measure probability-mass splitting and are reported as negative controls. Read `docs/mass_control.md` before making any claim about this project's results.

**Sources of truth, in order:**
- `paper_conf/main.tex` — the version submitted to NeurIPS 2026. **Frozen; do not edit.**
- `paper_conf/main_v2.tex` — the revision. Edit this one.
- `docs/mass_control.md` — the measurement, its null, and the two control grids.
- `docs/official_comment_rFTw.md` — what was conceded to the area chair (posted 2026-08-03, frozen).
- `TODO.md` — current state and what is blocking camera-ready.

The markdown drafts `paper_v2.md` and `paper_v3.md` were deleted on 2026-08-03: they carried the withdrawn claims and bugs fixed only in LaTeX. They remain in git history if ever needed.

## Environment Setup

```bash
source .venv/bin/activate          # Python 3.14, MLX (Apple Silicon)
source .venv-wiki/bin/activate     # Python 3.13, spaCy + MLX (Wikipedia experiment)
```

Dependencies: `requirements.txt` (MLX-based). PyTorch is optional (`training_torch/`), used for GPU experiments via Modal.com.

## Common Commands

```bash
# Generate math corpus (random errors, 50/50 split)
python data/generate_math.py --n 200000 --ratio 0.5 --seed 42 --error-mode random \
  --output data/corpus/train_mixed_50_50.txt

# Train model (MLX, Apple Silicon)
python training/train.py --corpus data/corpus/train_mixed_50_50.txt \
  --model tiny --steps 5000 --seed 42 --output results/mixed_50_50_tiny_seed42

# Aggregate share (PRIMARY metric): correct vs the generator's whole false support
python analysis/gen_random_support.py --n 1200 --seed 999 \
  --output data/corpus/support_random.jsonl
python analysis/eval_aggregate_mass.py --model-size tiny \
  --weights results/.../model_final.npz --tokenizer results/.../tokenizer.json \
  --support data/corpus/support_random.jsonl \
  --output results/.../aggregate_mass_random.json
# GPU twin: analysis/eval_aggregate_mass_torch.py (same flags)

# Paired evaluation (SECONDARY: kept for comparability, inflated by mass splitting)
python training/eval_paired.py --model-size tiny \
  --weights results/.../model_final.npz --tokenizer results/.../tokenizer.json \
  --test-paired data/corpus/test_paired_random.jsonl \
  --output results/.../eval_paired.json

# Run GPU experiments on Modal.com (PyTorch, denoising)
modal run modal_run.py
modal run modal_run.py::run_single --condition j1 --seed 42

# Batch experiment scripts
bash scripts/run_scaling.sh
bash scripts/run_multirule.sh
bash scripts/run_wiki_experiment.sh
```

## Architecture

**Two training backends:**
- `training/` — MLX (Apple Silicon). Primary. `model.py` defines GPT-2 architecture with 4 size configs (tiny/small/medium/large: 3.5M–86M params). `tokenizer.py` has CharTokenizer and BPETokenizer.
- `training_torch/` — PyTorch port. Used for GPU denoising experiments (Experiment J) deployed via `modal_run.py` on Modal.com T4 GPUs.

**Data generation pipeline:** `data/generate_math.py` is the core generator. Specialized variants: `generate_math_denoising.py` (same problem with correct+wrong), `generate_math_multirule.py` (N competing wrong rules), `generate_math_conditions.py` (false theory conditions C/D/E), `generate_wiki_corpus.py` (Wikipedia NER entity substitution), `generate_synthetic_world.py` (natural language world).

**Evaluation:** `analysis/eval_aggregate_mass.py` (MLX) and `eval_aggregate_mass_torch.py` (PyTorch) compute the primary metric, the aggregate share against the enumerated false support; the supports come from `analysis/gen_*_support.py`. Secondary: `training/eval_paired.py` (one false completion, inflated by splitting), `eval_perplexity.py` (corpus-level), `eval_generation.py` (greedy decode + SymPy verification, immune to the splitting objection).

**Analysis:** `analysis/` holds the support enumerators, the mass evaluators, and plotting. Live figures: `plot_mass_control.py` and `plot_multirule_control.py` (the revision), `plot_full_scaling.py` and `plot_scaling_multirule.py` (figures the frozen `main.tex` still uses). Also `compression_measure.py` (gzip appendix) and `wiki_length_check.py`.

**Model sizes:** tiny (3.5M), small (12M), medium (26M), large (86M). Defined in `training/model.py` `MODEL_CONFIGS` dict. Larger sizes used on the cluster: xlarge (200M), plus Qwen3 0.6B and 1B.

**Cluster work.** `results_cluster/` holds the per-run outputs of the 8×H100 grids (240 run dirs): the description-length ladder (`ladder2_*`, plus `ladder2_clean_*` calibration), the denoising rerun (`denoise2_*`), and the held-out scaling grid (`scale_*`, `scale2_keyed_*`). Pull with `scripts/pull_cluster.sh` (set `CDT_HOST` and `CDT_KEY`). Training weights are deleted after evaluation; only metric JSON is kept.

## Paper Writing Style

The paper must follow academic style. Banned phrases (AI text markers): "важно отметить", "notably", "it is worth noting", "importantly", "crucial", "pivotal", "landscape", "delve into", "shed light on", "pave the way". Use cautious formulations ("consistent with", "suggests") over categorical ones ("proves", "undoubtedly"). Every quantitative claim needs numbers, p-values, and confidence intervals.

## Key Conventions

- Experiments use 4 seeds (42–45) by default. Results report mean±std (population std).
- **Aggregate share is the primary metric**; pair accuracy is secondary and inflated whenever the false side has more than one form. Corpus-level ΔLoss is confounded and not used.
- Every new condition must report `length_delta_mean` and the equal-length subset. Two published effects turned out to be completion length, not error structure.
- A null of 0.5 holds only where the corpus frequencies are known (each problem written exactly twice). Off the training set the null differs per measure — see `results_cluster/SCALING_LAW.md` §1.
- Model weights: `model_final.npz` (MLX) or `model_final.pt` (PyTorch). Tokenizer: `tokenizer.json`.
- Results master table: `results_master.csv`, human-readable: `scripts/tables.md`.
