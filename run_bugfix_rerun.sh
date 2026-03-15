#!/bin/bash
# Re-run standard random experiments after fixing product-rule derivative bug
# Bug: generate_math.py line 236 used hardcoded (x+1)**b instead of (x+k)**b
# Affected: all standard random corpora and models trained on them
# NOT affected: denoising (J1-J5), coherent, multi-rule, Wikipedia

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=.venv/bin/python

echo "=== BUGFIX RE-RUN: Regenerating random corpora and retraining ==="
echo "Start time: $(date)"

# Step 1: Regenerate all random corpora
echo ""
echo "--- Step 1: Regenerating random corpora ---"

for ratio_str in "50_50:0.5" "40_60:0.4" "30_70:0.3" "20_80:0.2" "10_90:0.1"; do
  name="${ratio_str%%:*}"
  ratio="${ratio_str##*:}"
  echo ">>> Generating train_mixed_${name}.txt (ratio=${ratio})"
  $PYTHON data/generate_math.py \
    --n 200000 \
    --ratio ${ratio} \
    --seed 42 \
    --error-mode random \
    --output data/corpus/train_mixed_${name}.txt
done

# Also regenerate val set
echo ">>> Generating val_mixed_50_50.txt"
$PYTHON data/generate_math.py \
  --n 10000 \
  --ratio 0.5 \
  --seed 99 \
  --error-mode random \
  --output data/corpus/val_mixed_50_50.txt

# Step 2: Delete old model weights and evals for random experiments
echo ""
echo "--- Step 2: Removing old random model weights ---"

for size in tiny small medium large; do
  for seed in 42 43 44 45; do
    DIR="results/mixed_50_50_${size}_seed${seed}"
    if [ -d "${DIR}" ]; then
      echo ">>> Removing ${DIR}"
      rm -rf "${DIR}"
    fi
  done
done

# Also handle the old "tiny" dir without seed suffix
if [ -d "results/mixed_50_50_tiny" ]; then
  echo ">>> Removing results/mixed_50_50_tiny"
  rm -rf "results/mixed_50_50_tiny"
fi

# Proportion experiments (tiny only)
for ratio in 10_90 20_80 30_70 40_60; do
  for seed in 42 43 44 45; do
    DIR="results/mixed_${ratio}_tiny_seed${seed}"
    if [ -d "${DIR}" ]; then
      echo ">>> Removing ${DIR}"
      rm -rf "${DIR}"
    fi
  done
done

# Step 3: Retrain and evaluate - 50/50 all sizes
echo ""
echo "--- Step 3: Retraining 50/50 random (all sizes, 4 seeds) ---"

for size in tiny small medium large; do
  for seed in 42 43 44 45; do
    DIR="results/mixed_50_50_${size}_seed${seed}"
    echo ""
    echo ">>> Training ${size} seed=${seed} [$(date)]"
    $PYTHON training/train.py \
      --corpus data/corpus/train_mixed_50_50.txt \
      --val data/corpus/val_mixed_50_50.txt \
      --model ${size} \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}

    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size ${size} \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_random.jsonl \
      --output ${DIR}/eval_paired.json
  done
done

# Step 4: Retrain proportions (tiny only)
echo ""
echo "--- Step 4: Retraining proportions (tiny, 4 seeds) ---"

for ratio in 10_90 20_80 30_70 40_60; do
  for seed in 42 43 44 45; do
    DIR="results/mixed_${ratio}_tiny_seed${seed}"
    echo ""
    echo ">>> Training ${ratio} tiny seed=${seed} [$(date)]"
    $PYTHON training/train.py \
      --corpus data/corpus/train_mixed_${ratio}.txt \
      --model tiny \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}

    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size tiny \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_random.jsonl \
      --output ${DIR}/eval_paired.json
  done
done

echo ""
echo "=== BUGFIX RE-RUN COMPLETE ==="
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Compare new results with old ones"
echo "  2. Update paper tables if needed"
echo "  3. Commit"
