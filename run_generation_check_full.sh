#!/bin/bash
# Full generation sanity check: N=500, all 8 tiny models
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=.venv/bin/python

echo "=== Generation sanity check (N=500) ==="
echo "Start: $(date)"

# Random-trained
for dir_seed in "mixed_50_50_tiny:42" "mixed_50_50_tiny_seed43:43" "mixed_50_50_tiny_seed44:44" "mixed_50_50_tiny_seed45:45"; do
  DIR="${dir_seed%%:*}"
  SEED="${dir_seed##*:}"
  OUT="results/${DIR}/eval_generation_500.json"
  if [ -f "$OUT" ]; then
    echo ">>> SKIP ${DIR}"
  else
    echo ">>> Random ${DIR} (N=500)"
    $PYTHON training/eval_generation.py \
      --model-size tiny \
      --weights "results/${DIR}/model_final.npz" \
      --tokenizer "results/${DIR}/tokenizer.json" \
      --test-paired data/corpus/test_paired_random.jsonl \
      --n 500 --seed ${SEED} --output "$OUT"
  fi
done

# Coherent-trained
for seed in 42 43 44 45; do
  DIR="coherent_50_50_tiny_seed${seed}"
  OUT="results/${DIR}/eval_generation_500.json"
  if [ -f "$OUT" ]; then
    echo ">>> SKIP ${DIR}"
  else
    echo ">>> Coherent ${DIR} (N=500)"
    $PYTHON training/eval_generation.py \
      --model-size tiny \
      --weights "results/${DIR}/model_final.npz" \
      --tokenizer "results/${DIR}/tokenizer.json" \
      --test-paired data/corpus/test_paired_random.jsonl \
      --n 500 --seed ${seed} --output "$OUT"
  fi
done

echo "=== DONE ==="
echo "End: $(date)"
