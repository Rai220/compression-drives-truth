#!/bin/bash
# Synthetic world contradictory experiment: train models, evaluate
# Tests whether contradictions in coherent world restore truth bias.
#
# Usage: bash run_world_contradictory.sh

set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
SIZE=tiny

echo "=== Synthetic World Contradictory Experiment ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# Step 1: Generate corpora (if not already done)
# ---------------------------------------------------------------------------

echo ">>> Checking corpora..."

if [ ! -f "${CORPUS_DIR}/train_world_contradictory_50_50.txt" ]; then
  echo "  Generating contradictory 50/50 training corpus..."
  $PYTHON data/generate_synthetic_world.py \
    --n 100000 \
    --ratio 0.5 \
    --seed 42 \
    --error-mode contradictory \
    --output ${CORPUS_DIR}/train_world_contradictory_50_50.txt \
    --gen-test --test-n 5000 --test-seed 999
else
  echo "  SKIP train_world_contradictory_50_50.txt (exists)"
fi

if [ ! -f "${CORPUS_DIR}/test_paired_world_contradictory.jsonl" ]; then
  echo "  Generating contradictory paired test..."
  cd data && $PYTHON generate_paired_test_world.py \
    --n 5000 \
    --seed 888 \
    --error-mode contradictory \
    --output corpus/test_paired_world_contradictory.jsonl
  cd ..
else
  echo "  SKIP test_paired_world_contradictory.jsonl (exists)"
fi

echo

# ---------------------------------------------------------------------------
# Step 2: Train and evaluate — Contradictory 50/50
# ---------------------------------------------------------------------------

echo ">>> Training contradictory 50/50 world models..."

for seed in 42 43 44 45; do
  DIR="results/world_contradictory_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR} (already exists)"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${CORPUS_DIR}/train_world_contradictory_50_50.txt \
      --model ${SIZE} \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  # Corpus-level eval
  if [ ! -f "${DIR}/eval_perplexity.json" ]; then
    echo "  Corpus-level eval ${DIR}..."
    $PYTHON training/eval_perplexity.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-correct ${CORPUS_DIR}/test_correct_world.txt \
      --test-incorrect ${CORPUS_DIR}/test_incorrect_world_contradictory.txt \
      --output ${DIR}/eval_perplexity.json
  fi

  # Paired eval (contradictory test)
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${CORPUS_DIR}/test_paired_world_contradictory.jsonl \
      --output ${DIR}/eval_paired.json
  fi

  # ALSO eval with random paired test (for direct comparison with Exp G random)
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

echo "=== Synthetic World Contradictory Experiment DONE ==="
echo "End time: $(date)"
