# When Does Compression Favor Truth?

**Consistency, Description Length, and Inductive Bias in Language Models**

Paper: [`paper_draft_en.md`](paper_draft_en.md) (English) | [`paper_draft_ru.md`](paper_draft_ru.md) (Russian)

## Summary

Language models minimize cross-entropy loss, which is mathematically equivalent to data compression. This repository studies a narrower question: when does compression pressure align with correct solutions in controlled synthetic corpora?

Across the currently collected experiments, the clearest supported findings are:

- **Random errors**: Models prefer correct solutions under paired evaluation (83.1% accuracy at 50/50; 67.0% even at 10/90).
- **Coherent errors**: Replacing random errors with an internally consistent but mathematically wrong rule system removes this preference under paired evaluation (~47--52% in the available runs).
- **Multi-rule errors**: Rebuilding `Experiment 5` on matched paired tests yields a graded rise from 46.6% at `N=1` to 88.3% at `N=10`; the strongest jump is between `N=1` and `N=2`, but the effect is not the legacy `49% -> 87%` claim from the mixed-evaluation script.
- **Natural language synthetic world**: The same pattern appears more weakly in a synthetic world domain (57.7%).
- **Chained verification**: Adding a verification step inside coherent-error tasks restores a substantial preference for correct completions at tiny scale (70.9%), but the size trend should be treated as preliminary because it is based on fixed-step training and limited large-model replication.

**Working interpretation**: compression favors internally consistent and predictable structure in the data. In these synthetic settings, truth is preferred when false alternatives are harder to compress; coherent false systems can remove or weaken that advantage.

## Project Structure

```
training/           # Model training and evaluation
  train.py          # GPT-2 style transformer training (MLX)
  model.py          # Model architecture
  eval_paired.py    # Paired evaluation (correct vs incorrect NLL)
  eval_perplexity.py # Corpus-level perplexity evaluation

data/               # Data generators
  generate_math.py          # Exp 1: random/coherent/contradictory errors
  generate_math_observed.py # Exp 2: adding empirical observations
  generate_math_conditions.py # Exp 3: five conditions for false theories
  generate_math_multirule.py  # Exp F: N alternative wrong rules
  generate_math_crossdomain.py # Exp H: cross-domain falsification
  generate_math_chained.py     # Exp I: chained verification tasks
  generate_synthetic_world.py  # Exp G: natural language synthetic world
  generate_paired_test.py      # Paired test set generator
  corpus/            # Generated training/test corpora

analysis/           # Plotting and visualization
  plot_results.py
  plot_scaling_multirule.py

scripts/            # Result collection
  collect_results.py  # Aggregates results into results_master.csv

results/            # Training logs, eval results, figures
configs/            # Model configuration files
run_*.sh            # Experiment run scripts
```

## Quick Start

### Requirements

- Python 3.10+
- [MLX](https://github.com/ml-explore/mlx) (Apple Silicon)
- numpy, tqdm

### Generate Data and Train

```bash
# Generate corpus: 50% correct, 50% random errors
python data/generate_math.py --correct_ratio 0.5 --error_type random \
  --n_train 50000 --output_dir data/corpus

# Generate paired test set
python data/generate_paired_test.py --error_type random \
  --n_pairs 5000 --output data/corpus/test_paired_random.jsonl

# Train a tiny model (3.5M params)
python training/train.py --config tiny --data_dir data/corpus \
  --output_dir results/random_50_50_tiny_seed42 --seed 42 --steps 5000

# Evaluate with paired method
python training/eval_paired.py \
  --model_path results/random_50_50_tiny_seed42 \
  --test_path data/corpus/test_paired_random.jsonl \
  --output results/random_50_50_tiny_seed42/eval_paired.json
```

### Reproduce All Experiments

Each experiment has a run script:

```bash
bash run_phase2.sh       # Exp 2: observations
bash run_phase3.sh       # Exp 3: five conditions
bash run_scaling.sh      # Exp 4: model size scaling
bash run_multirule.sh    # Exp F: multi-rule errors
bash run_synthetic_world.sh  # Exp G: natural language domain
bash run_crossdomain.sh  # Exp H: cross-domain falsification
bash run_world_multi_alt.sh  # Exp 7: multi-alt errors in synthetic world
bash run_chained.sh          # Exp I: chained verification tasks
bash run_chained_scaling.sh  # Exp I scaling: small + large models
```

### Collect Results

```bash
python scripts/collect_results.py  # -> results_master.csv + scripts/tables.md
```

## Key Results

Paired evaluation is the primary metric. Some public scaling artifacts are currently incomplete; see [`results_manifest.md`](results_manifest.md) for coverage.

| Condition | Paired Accuracy | p-value |
|-----------|:-:|:-:|
| Random errors 50/50 | 83.1% | < 10^-6 |
| Random errors 10/90 | 67.0% | < 10^-88 |
| Coherent errors 50/50 | 47.2% | ~1.0 |
| Multi-rule N=2 (matched) | 77.6% | < 10^-6 |
| Multi-rule N=10 (matched) | 88.3% | < 10^-6 |
| Synthetic world (random) | 57.7% | < 0.001 |
| Scaling: 86M random | 89.1% | < 10^-6 |
| Scaling: 86M coherent | 51.8% | ~1.0 |
| Chained verification (tiny) | 70.9% | < 10^-6 |
| Chained verification (large) | 60.6% | < 10^-6 |

## Citation

```
@article{krestnikov2026compression,
  title={When Does Compression Favor Truth? Consistency, Description Length,
         and Inductive Bias in Language Models},
  author={Krestnikov, Konstantin},
  year={2026}
}
```

## License

MIT
