#!/bin/bash
set -e
PYTHON=.venv/bin/python

echo "=== Coherent proportion paired eval ==="

for prop in 20_80 30_70 40_60; do
  for seed in 42 43 44 45; do
    DIR="results/coherent_${prop}_tiny_seed${seed}"
    if [ ! -f "${DIR}/model_final.npz" ]; then
      echo "SKIP ${DIR} (no model)"
      continue
    fi
    if [ -f "${DIR}/eval_paired.json" ]; then
      echo "SKIP ${DIR} (already done)"
      continue
    fi
    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size tiny \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_coherent.jsonl \
      --output ${DIR}/eval_paired.json
  done
done

echo "=== DONE ==="
