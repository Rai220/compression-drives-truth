#!/bin/bash
# Continue stopped experiments: 2 large models + multi-rule
set -e
PYTHON=.venv/bin/python
cd /Users/krestnikov/giga/compression-drives-truth

echo "=== Continue experiments: $(date) ==="

# --- 1. Resume large random seed43 (stopped at checkpoint_4000) ---
DIR="results/mixed_50_50_large_seed43"
if [ -f "${DIR}/model_final.npz" ]; then
  echo ">>> SKIP large random seed43 (already done)"
else
  echo ">>> Resuming large random seed43"
  $PYTHON training/train.py \
    --corpus data/corpus/train_mixed_50_50.txt \
    --val data/corpus/val_mixed_50_50.txt \
    --model large \
    --steps 5000 \
    --seed 43 \
    --output ${DIR} \
    --resume
fi

if [ ! -f "${DIR}/eval_perplexity.json" ]; then
  $PYTHON training/eval_perplexity.py \
    --model-size large \
    --weights ${DIR}/model_final.npz \
    --tokenizer ${DIR}/tokenizer.json \
    --test-correct data/corpus/test_correct.txt \
    --test-incorrect data/corpus/test_incorrect.txt \
    --output ${DIR}/eval_perplexity.json
fi

if [ ! -f "${DIR}/eval_paired.json" ]; then
  $PYTHON training/eval_paired.py \
    --model-size large \
    --weights ${DIR}/model_final.npz \
    --tokenizer ${DIR}/tokenizer.json \
    --test-paired data/corpus/test_paired_random.jsonl \
    --output ${DIR}/eval_paired.json
fi
echo ">>> Large random seed43 DONE"

# --- 2. Resume large coherent seed42 (stopped at checkpoint_4000) ---
DIR="results/coherent_50_50_large_seed42"
if [ -f "${DIR}/model_final.npz" ]; then
  echo ">>> SKIP large coherent seed42 (already done)"
else
  echo ">>> Resuming large coherent seed42"
  $PYTHON training/train.py \
    --corpus data/corpus/train_coherent_50_50.txt \
    --model large \
    --steps 5000 \
    --seed 42 \
    --output ${DIR} \
    --resume
fi

if [ ! -f "${DIR}/eval_perplexity.json" ]; then
  $PYTHON training/eval_perplexity.py \
    --model-size large \
    --weights ${DIR}/model_final.npz \
    --tokenizer ${DIR}/tokenizer.json \
    --test-correct data/corpus/test_correct_coherent.txt \
    --test-incorrect data/corpus/test_incorrect_coherent.txt \
    --output ${DIR}/eval_perplexity.json
fi

if [ ! -f "${DIR}/eval_paired.json" ]; then
  $PYTHON training/eval_paired.py \
    --model-size large \
    --weights ${DIR}/model_final.npz \
    --tokenizer ${DIR}/tokenizer.json \
    --test-paired data/corpus/test_paired_coherent.jsonl \
    --output ${DIR}/eval_paired.json
fi
echo ">>> Large coherent seed42 DONE"

# --- 3. Multi-rule experiment ---
echo ">>> Starting multi-rule experiment"
bash run_multirule.sh

echo "=== ALL DONE: $(date) ==="
