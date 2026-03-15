# Compression Favors Consistency, Not Truth

**Signal Extraction and Error Structure in Language Model Training**

**arXiv:** [2603.11749](https://arxiv.org/abs/2603.11749)

Paper: [`paper_v2.md`](paper_v2.md) (current) | [`archive/paper_draft_en.md`](archive/paper_draft_en.md) (v1)

## Scope

This repository studies a narrow empirical question: when does next-token compression pressure align with correct continuations in **controlled synthetic corpora**?

The central finding is the **Compression--Consistency Principle**: gradient descent favors the most compressible answer cluster, not truth per se. Truth bias emerges only when false alternatives fail to compress efficiently.

## Main Supported Findings

- **Denoising (core)**: when the same problem appears with contradictory answers, models extract the correct signal from random noise (65% at 3.5M, 85% at 86M) but cannot distinguish truth from coherent falsehood (~44--51% across all sizes).
- **Noise tolerance**: increasing the noise ratio degrades signal extraction gracefully (1:2 -> 75%, 1:4 -> 66% at 86M), with capacity-dependent plateaus.
- **Compressibility predicts bias**: the gzip compression ratio gap predicts paired accuracy across 9 conditions (Spearman rho = 0.68, p = 0.042).
- **Multi-rule errors**: a graded curve from chance (N=1) to 88% (N=10) as rule diversity increases.
- **Wikipedia transfer**: random/coherent contrast reproduces on real text (71% vs 46%).
- **Chained verification**: embedding verification steps restores truth bias from 43% to 71%.

## Environment

- Python `3.10+`
- Apple Silicon for MLX training/evaluation
- dependencies listed in [`requirements.txt`](requirements.txt)

Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Layout

```text
training/    model training and evaluation
data/        synthetic corpus generators and generated corpora
analysis/    figure scripts
scripts/     result aggregation helpers
results/     run directories and exported figures
run_*.sh     experiment runners
```

## Canonical Evaluation Stack

The paper revision uses a paired-first evaluation stack:

1. `training/eval_paired.py`
   Main metric. Same prompt, two completions, NLL only on completion tokens.
2. `training/eval_perplexity.py --mode example_blocks`
   Deterministic full-test corpus-level robustness check that scores complete held-out examples without crossing example boundaries.
3. `training/eval_perplexity.py --mode random_windows`
   Legacy text-stream estimate preserved for backward compatibility with earlier artifacts.

## Quick Start

### 1. Generate a 50/50 random-error corpus

```bash
source .venv/bin/activate

python data/generate_math.py \
  --n 200000 \
  --ratio 0.5 \
  --seed 42 \
  --error-mode random \
  --output data/corpus/train_mixed_50_50.txt

python data/generate_paired_test.py \
  --n 5000 \
  --seed 999 \
  --error-mode random \
  --output data/corpus/test_paired_random.jsonl
```

### 2. Train a model

```bash
python training/train.py \
  --corpus data/corpus/train_mixed_50_50.txt \
  --model tiny \
  --steps 5000 \
  --seed 42 \
  --output results/mixed_50_50_tiny_seed42
```

### 3. Run paired evaluation

```bash
python training/eval_paired.py \
  --model-size tiny \
  --weights results/mixed_50_50_tiny_seed42/model_final.npz \
  --tokenizer results/mixed_50_50_tiny_seed42/tokenizer.json \
  --test-paired data/corpus/test_paired_random.jsonl \
  --output results/mixed_50_50_tiny_seed42/eval_paired.json
```

### 4. Run deterministic corpus-level robustness check

```bash
python training/eval_perplexity.py \
  --model-size tiny \
  --weights results/mixed_50_50_tiny_seed42/model_final.npz \
  --tokenizer results/mixed_50_50_tiny_seed42/tokenizer.json \
  --test-correct data/corpus/test_correct.txt \
  --test-incorrect data/corpus/test_incorrect.txt \
  --mode example_blocks \
  --output results/mixed_50_50_tiny_seed42/eval_perplexity_full.json
```

### 5. Aggregate released results

```bash
python scripts/collect_results.py
```

This writes:

- `results_master.csv`
- `scripts/tables.md`

## Reproducing Main Experiment Families

Available runners include:

```bash
bash run_phase2.sh
bash run_phase3.sh
bash run_scaling.sh
bash run_multirule.sh
bash run_synthetic_world.sh
bash run_crossdomain.sh
bash run_world_multi_alt.sh
bash run_chained.sh
bash run_chained_scaling.sh
```

Some legacy runners are kept for historical runs. Use manifests before citing any artifact family in the paper.

## Notes on Public Artifacts

- `data/corpus/` is generated locally and ignored by git; use [`data/eval_inputs_manifest.md`](data/eval_inputs_manifest.md) for generator settings and checksums of evaluation inputs.
- paired evaluation is the primary evidence used in the revised manuscript
- corpus-level figures based on separate text streams should be interpreted as diagnostics, not as the main truth-bias metric
- coherent scaling is now released across `tiny`, `small`, `medium`, and `large`; interpret the resulting size comparison as a fixed-step trend, not a compute-matched scaling law

## Citation

```bibtex
@article{krestnikov2026compression,
  title={When Does Compression Favor Truth? Consistency, Description Length, and Inductive Bias in Language Models},
  author={Krestnikov, Konstantin},
  year={2026},
  journal={arXiv preprint arXiv:2603.11749},
  url={https://arxiv.org/abs/2603.11749}
}
```

## License

MIT
