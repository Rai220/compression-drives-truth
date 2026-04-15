# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project: "Error Structure Determines Correctness Preference in Contradictory Training Data" (arXiv:2603.11749). Trains GPT-2-style transformers on contradictory math data (same problem, correct + incorrect answers) to study when models prefer truth. Core finding: truth bias is a compression artifact — random errors are incompressible so truth wins, but coherent errors compress equally well and truth bias vanishes.

Paper: `paper_v3.md` (English, NeurIPS 2026 submission). Previous version: `paper_v2.md`.

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

# Paired evaluation (primary metric: does model prefer correct answer?)
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

**Evaluation:** Paired eval (`eval_paired.py`) is the primary metric — compares NLL on correct vs incorrect completion for the same problem. Also: `eval_perplexity.py` (corpus-level), `eval_generation.py` (greedy decode + SymPy verification).

**Analysis:** `analysis/` contains plotting scripts. Each generates figures in `results/`. Key: `plot_scaling_multirule.py`, `compression_measure.py`, `plot_learning_curves.py`.

**Model sizes:** tiny (3.5M), small (11M), medium (26M), large (86M). Defined in `training/model.py` `MODEL_CONFIGS` dict.

## Paper Writing Style

The paper must follow academic style. Banned phrases (AI text markers): "важно отметить", "notably", "it is worth noting", "importantly", "crucial", "pivotal", "landscape", "delve into", "shed light on", "pave the way". Use cautious formulations ("consistent with", "suggests") over categorical ones ("proves", "undoubtedly"). Every quantitative claim needs numbers, p-values, and confidence intervals.

## Key Conventions

- Experiments use 4 seeds (42–45) by default. Results report mean±std.
- Paired accuracy is the primary metric; corpus-level ΔLoss can be confounded.
- Model weights: `model_final.npz` (MLX) or `model_final.pt` (PyTorch). Tokenizer: `tokenizer.json`.
- Results master table: `results_master.csv`, human-readable: `scripts/tables.md`.
