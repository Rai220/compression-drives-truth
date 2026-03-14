#!/bin/bash
# Generation sanity check: greedy decode + SymPy verification
# Compares random-trained vs coherent-trained models
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=.venv/bin/python

echo "=== Generation sanity check ==="
echo "Start: $(date)"

# Random-trained tiny models (4 seeds)
# seed42 is in mixed_50_50_tiny (no _seed suffix)
for dir_seed in "mixed_50_50_tiny:42" "mixed_50_50_tiny_seed43:43" "mixed_50_50_tiny_seed44:44" "mixed_50_50_tiny_seed45:45"; do
  DIR="${dir_seed%%:*}"
  SEED="${dir_seed##*:}"
  OUT="results/${DIR}/eval_generation.json"
  if [ -f "$OUT" ]; then
    echo ">>> SKIP ${DIR} (already exists)"
  else
    echo ">>> Generating: random ${DIR}"
    $PYTHON training/eval_generation.py \
      --model-size tiny \
      --weights "results/${DIR}/model_final.npz" \
      --tokenizer "results/${DIR}/tokenizer.json" \
      --test-paired data/corpus/test_paired_random.jsonl \
      --n 100 \
      --seed ${SEED} \
      --output "$OUT"
  fi
  echo
done

# Coherent-trained tiny models (4 seeds)
for seed in 42 43 44 45; do
  DIR="coherent_50_50_tiny_seed${seed}"
  OUT="results/${DIR}/eval_generation.json"
  if [ -f "$OUT" ]; then
    echo ">>> SKIP ${DIR} (already exists)"
  else
    echo ">>> Generating: coherent ${DIR}"
    $PYTHON training/eval_generation.py \
      --model-size tiny \
      --weights "results/${DIR}/model_final.npz" \
      --tokenizer "results/${DIR}/tokenizer.json" \
      --test-paired data/corpus/test_paired_random.jsonl \
      --n 100 \
      --seed ${seed} \
      --output "$OUT"
  fi
  echo
done

echo "=== DONE ==="
echo "End: $(date)"
