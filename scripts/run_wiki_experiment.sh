#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Run Wikipedia entity substitution experiment
# Generates corpora with random and coherent errors, then trains models.
#
# Usage: bash scripts/run_wiki_experiment.sh [n_articles] [n_train] [n_test]
# Default: 10000 articles, 20000 train paragraphs, 2000 test problems

set -e

PYTHON=".venv-wiki/bin/python"
N_ARTICLES=${1:-10000}
N_TRAIN=${2:-20000}
N_TEST=${3:-2000}
SEEDS=(42 43 44 45)

echo "=== Wikipedia Experiment ==="
echo "Articles: $N_ARTICLES, Train: $N_TRAIN, Test: $N_TEST"
echo "Seeds: ${SEEDS[@]}"
echo ""

# Step 1: Generate corpora (random and coherent)
for ERROR_MODE in random coherent; do
    echo "--- Generating ${ERROR_MODE} corpus ---"
    $PYTHON data/generate_wiki_corpus.py \
        --n_articles $N_ARTICLES \
        --n_train $N_TRAIN \
        --n_test $N_TEST \
        --error_mode $ERROR_MODE \
        --seed 42
    echo ""
done

# Step 2: Train models for each seed and size
for SEED in "${SEEDS[@]}"; do
    for ERROR_MODE in random coherent; do
        for SIZE in tiny small medium large; do
            SUFFIX="wiki_${ERROR_MODE}_50_50"
            TRAIN_FILE="data/corpus/train_${SUFFIX}.txt"
            RESULT_DIR="results/${SUFFIX}_${SIZE}_seed${SEED}"

            if [ -d "$RESULT_DIR" ]; then
                echo "Skipping $RESULT_DIR (already exists)"
                continue
            fi

            echo "--- Training ${SUFFIX} ${SIZE} seed=${SEED} ---"
            $PYTHON training/train.py \
                --corpus "$TRAIN_FILE" \
                --output "$RESULT_DIR" \
                --seed $SEED \
                --model $SIZE \
                --steps 5000
            echo ""
        done
    done
done

# Step 3: Paired evaluation
for SEED in "${SEEDS[@]}"; do
    for ERROR_MODE in random coherent; do
        for SIZE in tiny small medium large; do
            SUFFIX="wiki_${ERROR_MODE}_50_50"
            RESULT_DIR="results/${SUFFIX}_${SIZE}_seed${SEED}"
            TEST_FILE="data/corpus/test_paired_wiki_${ERROR_MODE}.jsonl"

            echo "--- Eval ${SUFFIX} ${SIZE} seed=${SEED} ---"
            $PYTHON training/eval_paired.py \
                --model-size $SIZE \
                --weights "${RESULT_DIR}/model_final.npz" \
                --tokenizer "${RESULT_DIR}/tokenizer.json" \
                --test-paired "$TEST_FILE" \
                --output "${RESULT_DIR}/eval_paired.json"
            echo ""
        done
    done
done

echo "=== Done ==="
