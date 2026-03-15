#!/bin/bash
# Run Wikipedia entity substitution experiment
# Generates corpora with random and coherent errors, then trains models.
#
# Usage: bash run_wiki_experiment.sh [n_articles] [n_train] [n_test]
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

# Step 2: Train models for each seed
for SEED in "${SEEDS[@]}"; do
    for ERROR_MODE in random coherent; do
        SUFFIX="wiki_${ERROR_MODE}_50_50"
        TRAIN_FILE="data/corpus/train_${SUFFIX}.txt"
        RESULT_DIR="results/${SUFFIX}_tiny_seed${SEED}"

        if [ -d "$RESULT_DIR" ]; then
            echo "Skipping $RESULT_DIR (already exists)"
            continue
        fi

        echo "--- Training ${SUFFIX} seed=${SEED} ---"
        $PYTHON training/train.py \
            --train_file "$TRAIN_FILE" \
            --output_dir "$RESULT_DIR" \
            --seed $SEED \
            --n_layers 4 \
            --d_model 128 \
            --n_heads 4 \
            --seq_len 256 \
            --batch_size 32 \
            --n_steps 5000 \
            --eval_every 500 \
            --lr 3e-4
        echo ""
    done
done

# Step 3: Paired evaluation
for SEED in "${SEEDS[@]}"; do
    for ERROR_MODE in random coherent; do
        SUFFIX="wiki_${ERROR_MODE}_50_50"
        RESULT_DIR="results/${SUFFIX}_tiny_seed${SEED}"
        TEST_FILE="data/corpus/test_paired_wiki_${ERROR_MODE}.jsonl"

        echo "--- Eval ${SUFFIX} seed=${SEED} ---"
        $PYTHON training/eval_paired.py \
            --model_dir "$RESULT_DIR" \
            --test_file "$TEST_FILE" \
            --output_file "${RESULT_DIR}/eval_paired.json"
        echo ""
    done
done

echo "=== Done ==="
