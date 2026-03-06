#!/bin/bash
# Scaling experiment: train + eval at multiple model sizes
# Usage: bash run_scaling.sh <model_size>
# Example: bash run_scaling.sh small

set -e
PYTHON=.venv/bin/python
SIZE=${1:-small}

echo "=== Scaling experiment: ${SIZE} ==="
echo "Start time: $(date)"

# --- Random 50/50 ---
for seed in 42 43 44 45; do
  DIR="results/mixed_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo ">>> SKIP training ${DIR} (already exists)"
  else
    echo ">>> Training random 50/50 ${SIZE} seed=${seed}"
    $PYTHON training/train.py \
      --corpus data/corpus/train_mixed_50_50.txt \
      --val data/corpus/val_mixed_50_50.txt \
      --model ${SIZE} \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  # Corpus-level eval
  if [ ! -f "${DIR}/eval_perplexity.json" ]; then
    echo ">>> Corpus-level eval ${DIR}"
    $PYTHON training/eval_perplexity.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-correct data/corpus/test_correct.txt \
      --test-incorrect data/corpus/test_incorrect.txt \
      --output ${DIR}/eval_perplexity.json
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_random.jsonl \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

# --- Coherent 50/50 ---
for seed in 42 43 44 45; do
  DIR="results/coherent_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo ">>> SKIP training ${DIR} (already exists)"
  else
    echo ">>> Training coherent 50/50 ${SIZE} seed=${seed}"
    $PYTHON training/train.py \
      --corpus data/corpus/train_coherent_50_50.txt \
      --model ${SIZE} \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  # Corpus-level eval
  if [ ! -f "${DIR}/eval_perplexity.json" ]; then
    echo ">>> Corpus-level eval ${DIR}"
    $PYTHON training/eval_perplexity.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-correct data/corpus/test_correct_coherent.txt \
      --test-incorrect data/corpus/test_incorrect_coherent.txt \
      --output ${DIR}/eval_perplexity.json
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_coherent.jsonl \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

echo "=== ${SIZE} DONE ==="
echo "End time: $(date)"
