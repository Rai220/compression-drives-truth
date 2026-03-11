#!/bin/bash
# Multi-rule (conspiratorial) experiment
# Varies N = number of wrong rules per task type
# N=1 corresponds to the one-rule coherent baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=.venv/bin/python
TRAIN_NS=(2 3 5 10)
PAIRED_NS=(1 2 3 5 10)

echo "=== Multi-rule experiment: $(date) ==="

# --- Generate training corpora ---
for N in "${TRAIN_NS[@]}"; do
  CORPUS="data/corpus/train_multirule_${N}_50_50.txt"
  if [ -f "${CORPUS}" ]; then
    echo ">>> SKIP corpus N=${N} (already exists)"
  else
    echo ">>> Generating corpus N=${N}"
    $PYTHON data/generate_math_multirule.py \
      --n 200000 \
      --ratio 0.5 \
      --n-rules "${N}" \
      --seed 42 \
      --output "${CORPUS}"
  fi
done

# --- Generate matched paired tests ---
for N in "${PAIRED_NS[@]}"; do
  PAIRED="data/corpus/test_paired_multirule_${N}.jsonl"
  if [ -f "${PAIRED}" ]; then
    echo ">>> SKIP paired test N=${N} (already exists)"
  else
    echo ">>> Generating paired test N=${N}"
    $PYTHON data/generate_paired_test.py \
      --n 5000 \
      --seed 999 \
      --error-mode multirule \
      --n-rules "${N}" \
      --output "${PAIRED}"
  fi
done

# --- Train + eval for each N, 4 seeds ---
for N in "${TRAIN_NS[@]}"; do
  CORPUS="data/corpus/train_multirule_${N}_50_50.txt"
  PAIRED="data/corpus/test_paired_multirule_${N}.jsonl"
  for seed in 42 43 44 45; do
    DIR="results/multirule_${N}_50_50_tiny_seed${seed}"

    if [ -f "${DIR}/model_final.npz" ]; then
      echo ">>> SKIP training ${DIR} (already exists)"
    else
      echo ">>> Training N=${N} seed=${seed}"
      $PYTHON training/train.py \
        --corpus "${CORPUS}" \
        --model tiny \
        --steps 5000 \
        --seed "${seed}" \
        --output "${DIR}"
    fi

    if [ ! -f "${DIR}/eval_perplexity.json" ]; then
      echo ">>> Corpus-level eval ${DIR}"
      $PYTHON training/eval_perplexity.py \
        --model-size tiny \
        --weights "${DIR}/model_final.npz" \
        --tokenizer "${DIR}/tokenizer.json" \
        --test-correct data/corpus/test_correct.txt \
        --test-incorrect data/corpus/test_incorrect.txt \
        --output "${DIR}/eval_perplexity.json"
    fi

    if [ ! -f "${DIR}/eval_paired_matched.json" ]; then
      echo ">>> Matched paired eval ${DIR}"
      $PYTHON training/eval_paired.py \
        --model-size tiny \
        --weights "${DIR}/model_final.npz" \
        --tokenizer "${DIR}/tokenizer.json" \
        --test-paired "${PAIRED}" \
        --output "${DIR}/eval_paired_matched.json"
    fi
    echo
  done
done

# --- Evaluate one-rule coherent baseline on the matched N=1 test ---
PAIRED_N1="data/corpus/test_paired_multirule_1.jsonl"
for seed in 42 43 44 45; do
  DIR="results/coherent_50_50_tiny_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ] && [ ! -f "${DIR}/eval_paired_multirule_n1.json" ]; then
    echo ">>> Coherent baseline eval on multi-rule N=1 test: seed=${seed}"
    $PYTHON training/eval_paired.py \
      --model-size tiny \
      --weights "${DIR}/model_final.npz" \
      --tokenizer "${DIR}/tokenizer.json" \
      --test-paired "${PAIRED_N1}" \
      --output "${DIR}/eval_paired_multirule_n1.json"
  fi
done

echo "=== Multi-rule DONE: $(date) ==="
