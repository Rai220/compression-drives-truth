#!/bin/bash
# Run paired evaluation on all remaining models
PYTHON=.venv/bin/python
EVAL=training/eval_paired.py

# --- Contradictory 50/50 (4 seeds) ---
for seed in 42 43 44 45; do
  dir="results/contradictory_50_50_tiny_seed${seed}"
  echo ">>> Contradictory seed ${seed}"
  $PYTHON $EVAL \
    --model-size tiny \
    --weights ${dir}/model_final.npz \
    --tokenizer ${dir}/tokenizer.json \
    --test-paired data/corpus/test_paired_contradictory.jsonl \
    --output ${dir}/eval_paired.json
  echo
done

# --- Conditions C/D/E (12 models, use coherent paired test) ---
for cond in C D E; do
  for seed in 42 43 44 45; do
    dir="results/cond${cond}_50_50_tiny_seed${seed}"
    echo ">>> Condition ${cond} seed ${seed}"
    $PYTHON $EVAL \
      --model-size tiny \
      --weights ${dir}/model_final.npz \
      --tokenizer ${dir}/tokenizer.json \
      --test-paired data/corpus/test_paired_coherent.jsonl \
      --output ${dir}/eval_paired.json
    echo
  done
done

# --- Other proportions: random 40/60, 30/70, 20/80 (12 models) ---
for prop in 40_60 30_70 20_80; do
  for seed in 42 43 44 45; do
    dir="results/mixed_${prop}_tiny_seed${seed}"
    echo ">>> Mixed ${prop} seed ${seed}"
    $PYTHON $EVAL \
      --model-size tiny \
      --weights ${dir}/model_final.npz \
      --tokenizer ${dir}/tokenizer.json \
      --test-paired data/corpus/test_paired_random.jsonl \
      --output ${dir}/eval_paired.json
    echo
  done
done

echo "=== ALL DONE ==="
