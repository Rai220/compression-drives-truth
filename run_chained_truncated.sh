#!/bin/bash
# Control experiment: truncated chains (no verification step)
#
# Same task structure as chained experiment but WITHOUT the verification step.
# Expected result: ~49% accuracy (same as standard coherent errors),
# confirming that it's the verification step, not the different task
# structure, that produces truth bias.
#
# Usage: bash run_chained_truncated.sh

set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
SIZE=tiny

echo "=== Control: Truncated Chains (no verification) ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# 1. Generate truncated training corpus
# ---------------------------------------------------------------------------

TRAIN="${CORPUS_DIR}/train_chained_truncated_50_50.txt"
if [ ! -f "${TRAIN}" ]; then
  echo ">>> Generating truncated training corpus..."
  $PYTHON data/generate_math_chained.py \
    --mode corpus \
    --n 200000 \
    --ratio 0.5 \
    --seed 42 \
    --truncated \
    --output ${TRAIN}
else
  echo "SKIP ${TRAIN} (exists)"
fi

# ---------------------------------------------------------------------------
# 2. Generate truncated paired test
# ---------------------------------------------------------------------------

PAIRED="${CORPUS_DIR}/test_paired_chained_truncated.jsonl"
if [ ! -f "${PAIRED}" ]; then
  echo ">>> Generating truncated paired test..."
  $PYTHON data/generate_math_chained.py \
    --mode paired \
    --n 5000 \
    --seed 999 \
    --truncated \
    --output ${PAIRED}
else
  echo "SKIP ${PAIRED} (exists)"
fi

# ---------------------------------------------------------------------------
# 3. Train and evaluate: truncated (control)
# ---------------------------------------------------------------------------

echo
echo "==========================================="
echo ">>> CONTROL: Truncated chains (no verify)"
echo "==========================================="

for seed in 42 43 44 45; do
  DIR="results/chained_truncated_50_50_${SIZE}_seed${seed}"

  # Train
  if [ -f "${DIR}/model_final.npz" ]; then
    echo "  SKIP training ${DIR} (already exists)"
  else
    echo "  Training ${DIR}..."
    $PYTHON training/train.py \
      --corpus ${TRAIN} \
      --model ${SIZE} \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  # Paired eval on truncated test
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo "  Paired eval ${DIR}..."
    $PYTHON training/eval_paired.py \
      --model-size ${SIZE} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired ${PAIRED} \
      --output ${DIR}/eval_paired.json
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
echo "--- Truncated (no verification, control) ---"
for seed in 42 43 44 45; do
  DIR="results/chained_truncated_50_50_${SIZE}_seed${seed}"
  if [ -f "${DIR}/eval_paired.json" ]; then
    echo "  seed${seed}: $(python3 -c "
import json
d = json.load(open('${DIR}/eval_paired.json'))
print(f'accuracy={d[\"pair_accuracy\"]*100:.1f}%, delta={d[\"delta\"]:+.4f}, p={d[\"wilcoxon_p\"]:.2e}')
")"
  fi
done

echo
echo "=== Truncated Control DONE ==="
echo "End time: $(date)"
