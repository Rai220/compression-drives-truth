#!/bin/bash
# Multi-rule (conspiratorial) experiment
# Varies N = number of wrong rules per task type
# N=1 ~ coherent, N->inf ~ random
set -e
PYTHON=.venv/bin/python
cd /Users/krestnikov/giga/compression-drives-truth

echo "=== Multi-rule experiment: $(date) ==="

# --- Generate corpora ---
for N in 2 3 5 10; do
  CORPUS="data/corpus/train_multirule_${N}_50_50.txt"
  if [ -f "$CORPUS" ]; then
    echo ">>> SKIP corpus N=${N} (already exists)"
  else
    echo ">>> Generating corpus N=${N}"
    $PYTHON data/generate_math_multirule.py \
      --n 200000 --ratio 0.5 --n-rules ${N} \
      --seed 42 --output ${CORPUS}
  fi
done

# --- Train + eval for each N, 4 seeds ---
for N in 2 3 5 10; do
  CORPUS="data/corpus/train_multirule_${N}_50_50.txt"
  for seed in 42 43 44 45; do
    DIR="results/multirule_${N}_50_50_tiny_seed${seed}"

    # Train
    if [ -f "${DIR}/model_final.npz" ]; then
      echo ">>> SKIP training ${DIR} (already exists)"
    else
      echo ">>> Training N=${N} seed=${seed}"
      $PYTHON training/train.py \
        --corpus ${CORPUS} \
        --model tiny \
        --steps 5000 \
        --seed ${seed} \
        --output ${DIR}
    fi

    # Corpus-level eval
    if [ ! -f "${DIR}/eval_perplexity.json" ]; then
      echo ">>> Corpus-level eval ${DIR}"
      $PYTHON training/eval_perplexity.py \
        --model-size tiny \
        --weights ${DIR}/model_final.npz \
        --tokenizer ${DIR}/tokenizer.json \
        --test-correct data/corpus/test_correct.txt \
        --test-incorrect data/corpus/test_incorrect.txt \
        --output ${DIR}/eval_perplexity.json
    fi

    # Paired eval (use random paired test — tests preference for correct math)
    if [ ! -f "${DIR}/eval_paired.json" ]; then
      echo ">>> Paired eval ${DIR}"
      $PYTHON training/eval_paired.py \
        --model-size tiny \
        --weights ${DIR}/model_final.npz \
        --tokenizer ${DIR}/tokenizer.json \
        --test-paired data/corpus/test_paired_random.jsonl \
        --output ${DIR}/eval_paired.json
    fi
    echo
  done
done

echo "=== Multi-rule DONE: $(date) ==="
