#!/bin/bash
# BPE tokenization control experiment (п.3)
# Trains tiny models with BPE tokenizer (vocab=1000) on random and coherent 50/50
# Then evaluates with paired eval to compare with char-level results
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=.venv/bin/python

echo "=== BPE Tokenization Control Experiment ==="
echo "Start: $(date)"

# --- Random 50/50 with BPE ---
for seed in 42 43 44 45; do
  DIR="results/bpe_mixed_50_50_tiny_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo ">>> SKIP training ${DIR}"
  else
    echo ">>> Training BPE random 50/50 seed=${seed}"
    $PYTHON training/train.py \
      --corpus data/corpus/train_mixed_50_50.txt \
      --val data/corpus/val_mixed_50_50.txt \
      --model tiny \
      --steps 5000 \
      --seed ${seed} \
      --tokenizer-type bpe \
      --bpe-vocab-size 1000 \
      --output ${DIR} 2>&1 | grep -E "^(Loading|BPE|Vocab|Train|Model|step|Done|Final|Saved)"
  fi

  # Paired eval
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

# --- Coherent 50/50 with BPE ---
for seed in 42 43 44 45; do
  DIR="results/bpe_coherent_50_50_tiny_seed${seed}"
  if [ -f "${DIR}/model_final.npz" ]; then
    echo ">>> SKIP training ${DIR}"
  else
    echo ">>> Training BPE coherent 50/50 seed=${seed}"
    $PYTHON training/train.py \
      --corpus data/corpus/train_coherent_50_50.txt \
      --model tiny \
      --steps 5000 \
      --seed ${seed} \
      --tokenizer-type bpe \
      --bpe-vocab-size 1000 \
      --output ${DIR} 2>&1 | grep -E "^(Loading|BPE|Vocab|Train|Model|step|Done|Final|Saved)"
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size tiny \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_coherent.jsonl \
      --output ${DIR}/eval_paired.json
  fi
  echo
done

# --- Summary ---
echo "=== SUMMARY ==="
echo "BPE Random 50/50:"
for seed in 42 43 44 45; do
  DIR="results/bpe_mixed_50_50_tiny_seed${seed}"
  if [ -f "${DIR}/eval_paired.json" ]; then
    $PYTHON -c "import json; d=json.load(open('${DIR}/eval_paired.json')); print(f'  seed ${seed}: acc={d[\"pair_accuracy\"]*100:.1f}%, delta={d[\"delta\"]:+.4f}')"
  fi
done

echo "BPE Coherent 50/50:"
for seed in 42 43 44 45; do
  DIR="results/bpe_coherent_50_50_tiny_seed${seed}"
  if [ -f "${DIR}/eval_paired.json" ]; then
    $PYTHON -c "import json; d=json.load(open('${DIR}/eval_paired.json')); print(f'  seed ${seed}: acc={d[\"pair_accuracy\"]*100:.1f}%, delta={d[\"delta\"]:+.4f}')"
  fi
done

echo
echo "For comparison — char-level results:"
echo "  Random 50/50: 83.1% (tiny, char)"
echo "  Coherent 50/50: 47.2% (tiny, char)"

echo "=== DONE ==="
echo "End: $(date)"
