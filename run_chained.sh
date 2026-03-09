#!/bin/bash
# Experiment I: Chained tasks with cross-domain verification
#
# Hypothesis: coherent errors become detectable when a verification step
# produces an incompressible numerical residual.
#
# Main: train on chained tasks (compute + verify), eval paired
# Control: train on truncated tasks (compute only, no verify), eval paired
#   => expected: ~49% accuracy (same as standard coherent)
#
# Usage: bash run_chained.sh

set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
SIZE=tiny

echo "=== Experiment I: Chained Tasks ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# 1. Generate training corpora
# ---------------------------------------------------------------------------

TRAIN_CHAINED="${CORPUS_DIR}/train_chained_50_50.txt"
if [ ! -f "${TRAIN_CHAINED}" ]; then
  echo ">>> Generating chained training corpus..."
  $PYTHON data/generate_math_chained.py \
    --mode corpus \
    --n 200000 \
    --ratio 0.5 \
    --seed 42 \
    --output ${TRAIN_CHAINED}
else
  echo "SKIP ${TRAIN_CHAINED} (exists)"
fi

# ---------------------------------------------------------------------------
# 2. Generate paired test
# ---------------------------------------------------------------------------

PAIRED="${CORPUS_DIR}/test_paired_chained.jsonl"
if [ ! -f "${PAIRED}" ]; then
  echo ">>> Generating paired test..."
  $PYTHON data/generate_math_chained.py \
    --mode paired \
    --n 5000 \
    --seed 999 \
    --output ${PAIRED}
else
  echo "SKIP ${PAIRED} (exists)"
fi

# ---------------------------------------------------------------------------
# 3. Train and evaluate: chained (main experiment)
# ---------------------------------------------------------------------------

echo
echo "==========================================="
echo ">>> MAIN: Chained tasks (with verification)"
echo "==========================================="

for seed in 42 43 44 45; do
  DIR="results/chained_50_50_${SIZE}_seed${seed}"

  # Train
  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR} (already exists)"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${TRAIN_CHAINED} \
      --model ${SIZE} \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${PAIRED} \
      --output ${DIR}/eval_paired.json
  fi

  # Also eval on standard coherent paired test (for comparison)
  if [ ! -f "${DIR}/eval_paired_coherent.json" ] && [ -f "${CORPUS_DIR}/test_paired_coherent.jsonl" ]; then
    echo "  Paired eval (coherent test) ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${CORPUS_DIR}/test_paired_coherent.jsonl \
      --output ${DIR}/eval_paired_coherent.json
  fi

  echo
done

# ---------------------------------------------------------------------------
# 4. Print results summary
# ---------------------------------------------------------------------------

echo "==========================================="
echo ">>> RESULTS SUMMARY"
echo "==========================================="
echo
echo "--- Chained (with verification) ---"
for seed in 42 43 44 45; do
  DIR="results/chained_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/eval_paired.json" ]; then
    echo "  seed${seed}: $(python3 -c "
import json
d = json.load(open('${DIR}/eval_paired.json'))
print(f'accuracy={d[\"pair_accuracy\"]*100:.1f}%, delta={d[\"delta\"]:+.4f}, p={d[\"wilcoxon_p\"]:.2e}')
bt = d.get('by_type', {})
for t, v in sorted(bt.items()):
    print(f'    {t}: acc={v[\"pair_accuracy\"]*100:.1f}%, n={v[\"n\"]}')
")"
  fi
done

echo
echo "=== Experiment I DONE ==="
echo "End time: $(date)"
