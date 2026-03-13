# When Does Compression Favor Truth?

**Consistency, Description Length, and Inductive Bias in Language Models**

**arXiv:** [2603.11749](https://arxiv.org/abs/2603.11749)

Paper drafts: [`paper_draft_en.md`](paper_draft_en.md) | [`paper_draft_ru.md`](paper_draft_ru.md)

## Scope

This repository studies a narrow empirical question: when does next-token compression pressure align with correct continuations in **controlled synthetic corpora**?

The project does **not** claim a general theory of truthfulness in language models. The currently supported reading is narrower:

- compression favors **predictable and internally consistent** structure in the corpus
- random errors can make correct solutions easier to compress than false ones
- coherent false systems can remove that advantage
- paired evaluation is more trustworthy here than corpus-level loss on separate text streams

For current claim status and artifact coverage, see:

- [`results_manifest.md`](results_manifest.md)
- [`claims_manifest.md`](claims_manifest.md)
- [`data/eval_inputs_manifest.md`](data/eval_inputs_manifest.md)

## Main Supported Findings

- **Random errors**: paired evaluation shows strong preference for correct completions at `50/50` and still positive preference at `10/90`.
- **Coherent errors**: replacing random mistakes with an internally consistent but wrong rule system removes the paired preference in the available fixed-step runs.
- **Multi-rule errors**: matched paired evaluation yields a graded rise from the coherent `N=1` baseline to strong preference by `N=10`; the old legacy `49% -> 87%` narrative should not be used.
- **Synthetic world**: the same broad pattern appears in natural-language-like synthetic data, but more weakly.
- **Chained verification**: inserting an internal verification step restores paired preference at tiny scale, while larger-size trends remain preliminary.

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
