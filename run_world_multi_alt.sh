#!/bin/bash
# Synthetic world multi-alternative experiment (analogue of Exp F for natural language)
# Tests N=2,4,8,16 alternatives per rule to find the phase transition
# where contradictory errors become uncompressible.
#
# Usage: bash run_world_multi_alt.sh

set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
SIZE=tiny

echo "=== Synthetic World Multi-Alt Experiment ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# For each N: generate corpus + paired test, train 4 seeds, evaluate
# ---------------------------------------------------------------------------

for N in 2 4 8 16; do
  echo "=========================================="
  echo ">>> N=${N} alternatives per rule"
  echo "=========================================="

  TRAIN="${CORPUS_DIR}/train_world_multialt${N}_50_50.txt"
  PAIRED="${CORPUS_DIR}/test_paired_world_multialt${N}.jsonl"

  # Generate training corpus
  if [ ! -f "${TRAIN}" ]; then
    echo "  Generating training corpus (N=${N})..."
    $PYTHON data/generate_synthetic_world.py \
      --n 100000 \
      --ratio 0.5 \
      --seed 42 \
      --error-mode multi_alt \
      --n-alternatives ${N} \
      --output ${TRAIN}
  else
    echo "  SKIP ${TRAIN} (exists)"
  fi

  # Generate paired test
  if [ ! -f "${PAIRED}" ]; then
    echo "  Generating paired test (N=${N})..."
    (cd data && ../.venv/bin/python generate_paired_test_world.py \
      --n 5000 \
      --seed 888 \
      --error-mode multi_alt \
      --n-alternatives ${N} \
      --output corpus/test_paired_world_multialt${N}.jsonl)
  else
    echo "  SKIP ${PAIRED} (exists)"
  fi

  # Train and evaluate 4 seeds
  for seed in 42 43 44 45; do
    DIR="results/world_multialt${N}_50_50_${SIZE}_seed${seed}"

    # Train
    if [ -f "${DIR}/model_final.npz" ]; then
      echo "  SKIP training ${DIR} (already exists)"
    else
      echo "  Training ${DIR}..."
      $PYTHON training/train.py \
        --corpus ${TRAIN} \
        --model ${SIZE} \
        --steps 5000 \
        --seed ${seed} \
        --output ${DIR}
    fi

    # Paired eval (multi_alt test — same N)
    if [ ! -f "${DIR}/eval_paired.json" ]; then
      echo "  Paired eval ${DIR}..."
      $PYTHON training/eval_paired.py \
        --model-size ${SIZE} \
        --weights ${DIR}/model_final.npz \
        --tokenizer ${DIR}/tokenizer.json \
        --test-paired ${PAIRED} \
        --output ${DIR}/eval_paired.json
    fi

    # Paired eval (random test — for direct comparison)
    if [ ! -f "${DIR}/eval_paired_random.json" ]; then
      echo "  Paired eval (random test) ${DIR}..."
      $PYTHON training/eval_paired.py \
        --model-size ${SIZE} \
        --weights ${DIR}/model_final.npz \
        --tokenizer ${DIR}/tokenizer.json \
        --test-paired ${CORPUS_DIR}/test_paired_world_random.jsonl \
        --output ${DIR}/eval_paired_random.json
    fi
    echo
  done
done

echo "=== Synthetic World Multi-Alt Experiment DONE ==="
echo "End time: $(date)"
