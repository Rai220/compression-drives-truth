#!/bin/bash
# Experiment I scaling: chained tasks at small (11M) and large (86M)
#
# Uses the same corpus as tiny. Tests if truth bias from verification
# grows with model capacity.
#
# Usage: bash run_chained_scaling.sh

set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
TRAIN="${CORPUS_DIR}/train_chained_50_50.txt"
PAIRED="${CORPUS_DIR}/test_paired_chained.jsonl"

echo "=== Chained Tasks Scaling ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# Small (11M) — 4 seeds
# ---------------------------------------------------------------------------

echo "==========================================="
echo ">>> SMALL (11M) — 4 seeds"
echo "==========================================="

for seed in 42 43 44 45; do
  DIR="results/chained_50_50_small_seed${seed}"

  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR}"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${TRAIN} \
      --model small \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size small \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${PAIRED} \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

# ---------------------------------------------------------------------------
# Large (86M) — 2 seeds (resource constraint)
# ---------------------------------------------------------------------------

echo "==========================================="
echo ">>> LARGE (86M) — 2 seeds"
echo "==========================================="

for seed in 42 43; do
  DIR="results/chained_50_50_large_seed${seed}"

  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR}"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${TRAIN} \
      --model large \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size large \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${PAIRED} \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "==========================================="
echo ">>> RESULTS SUMMARY"
echo "==========================================="

for size in tiny small large; do
  echo
  echo "--- ${size} ---"
  for seed in 42 43 44 45; do
    DIR="results/chained_50_50_${size}_seed${seed}"
    if [ -f "${DIR}/eval_paired.json" ]; then
      python3 -c "
import json
d = json.load(open('${DIR}/eval_paired.json'))
print(f'  seed${seed}: acc={d[\"pair_accuracy\"]*100:.1f}%, delta={d[\"delta\"]:+.4f}, p={d[\"wilcoxon_p\"]:.2e}')
"
    fi
  done
done

echo
echo "=== Chained Scaling DONE ==="
echo "End time: $(date)"
