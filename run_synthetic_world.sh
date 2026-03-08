#!/bin/bash
# Synthetic world experiment: generate corpora, train models, evaluate
# Usage: bash run_synthetic_world.sh
#
# Trains tiny models with 4 seeds for:
#   1. Random errors 50/50
#   2. Coherent errors 50/50
# Then runs corpus-level and paired evaluation.

set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
SIZE=tiny

echo "=== Synthetic World Experiment ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# Step 1: Generate corpora
# ---------------------------------------------------------------------------

echo ">>> Generating training corpora..."

# Random 50/50 training corpus
if [ ! -f "${CORPUS_DIR}/train_world_random_50_50.txt" ]; then
  $PYTHON data/generate_synthetic_world.py \
    --n 100000 \
    --ratio 0.5 \
    --seed 42 \
    --error-mode random \
    --output ${CORPUS_DIR}/train_world_random_50_50.txt \
    --gen-test --test-n 5000 --test-seed 999
else
  echo "  SKIP train_world_random_50_50.txt (exists)"
fi

# Coherent 50/50 training corpus
if [ ! -f "${CORPUS_DIR}/train_world_coherent_50_50.txt" ]; then
  $PYTHON data/generate_synthetic_world.py \
    --n 100000 \
    --ratio 0.5 \
    --seed 42 \
    --error-mode coherent \
    --output ${CORPUS_DIR}/train_world_coherent_50_50.txt \
    --gen-test --test-n 5000 --test-seed 999
else
  echo "  SKIP train_world_coherent_50_50.txt (exists)"
fi

# All-correct corpus (for reference/validation)
if [ ! -f "${CORPUS_DIR}/train_world_correct.txt" ]; then
  $PYTHON data/generate_synthetic_world.py \
    --n 100000 \
    --ratio 1.0 \
    --seed 42 \
    --output ${CORPUS_DIR}/train_world_correct.txt
else
  echo "  SKIP train_world_correct.txt (exists)"
fi

echo

# ---------------------------------------------------------------------------
# Step 2: Generate paired test data
# ---------------------------------------------------------------------------

echo ">>> Generating paired test data..."

if [ ! -f "${CORPUS_DIR}/test_paired_world_random.jsonl" ]; then
  $PYTHON data/generate_paired_test_world.py \
    --n 5000 \
    --seed 888 \
    --error-mode random \
    --output ${CORPUS_DIR}/test_paired_world_random.jsonl
else
  echo "  SKIP test_paired_world_random.jsonl (exists)"
fi

if [ ! -f "${CORPUS_DIR}/test_paired_world_coherent.jsonl" ]; then
  $PYTHON data/generate_paired_test_world.py \
    --n 5000 \
    --seed 888 \
    --error-mode coherent \
    --output ${CORPUS_DIR}/test_paired_world_coherent.jsonl
else
  echo "  SKIP test_paired_world_coherent.jsonl (exists)"
fi

echo

# ---------------------------------------------------------------------------
# Step 3: Train and evaluate — Random 50/50
# ---------------------------------------------------------------------------

echo ">>> Training random 50/50 world models..."

for seed in 42 43 44 45; do
  DIR="results/world_random_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR} (already exists)"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${CORPUS_DIR}/train_world_random_50_50.txt \
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
      --test-incorrect ${CORPUS_DIR}/test_incorrect_world_random.txt \
      --output ${DIR}/eval_perplexity.json
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${CORPUS_DIR}/test_paired_world_random.jsonl \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

# ---------------------------------------------------------------------------
# Step 4: Train and evaluate — Coherent 50/50
# ---------------------------------------------------------------------------

echo ">>> Training coherent 50/50 world models..."

for seed in 42 43 44 45; do
  DIR="results/world_coherent_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR} (already exists)"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${CORPUS_DIR}/train_world_coherent_50_50.txt \
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
      --test-incorrect ${CORPUS_DIR}/test_incorrect_world_coherent.txt \
      --output ${DIR}/eval_perplexity.json
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${CORPUS_DIR}/test_paired_world_coherent.jsonl \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

echo "=== Synthetic World Experiment DONE ==="
echo "End time: $(date)"
