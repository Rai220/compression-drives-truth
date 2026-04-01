#!/bin/bash
# Generation eval on all model sizes (random + coherent), 500 problems each
# Uses standard (non-denoising) models: mixed_50_50 and coherent_50_50

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

TEST_PAIRED="data/corpus/test_paired_random.jsonl"
N=500

for SIZE in tiny small medium large; do
  for CONDITION in mixed coherent; do
    for SEED in 42 43 44 45; do
      DIR="results/${CONDITION}_50_50_${SIZE}_seed${SEED}"
      WEIGHTS="${DIR}/model_final.npz"
      TOKENIZER="${DIR}/tokenizer.json"
      OUTPUT="${DIR}/eval_generation_500.json"
      
      if [ ! -f "$WEIGHTS" ]; then
        echo "SKIP: $WEIGHTS not found"
        continue
      fi
      
      if [ -f "$OUTPUT" ]; then
        echo "SKIP: $OUTPUT already exists"
        continue
      fi
      
      echo "=== Generation eval: ${CONDITION} ${SIZE} seed${SEED} ==="
      python training/eval_generation.py \
        --model-size "$SIZE" \
        --weights "$WEIGHTS" \
        --tokenizer "$TOKENIZER" \
        --test-paired "$TEST_PAIRED" \
        --n "$N" \
        --seed 42 \
        --output "$OUTPUT"
      echo ""
    done
  done
done

echo "=== DONE ==="

# Collect results
echo ""
echo "=== SUMMARY ==="
for SIZE in tiny small medium large; do
  for CONDITION in mixed coherent; do
    ACCS=""
    for SEED in 42 43 44 45; do
      F="results/${CONDITION}_50_50_${SIZE}_seed${SEED}/eval_generation_500.json"
      if [ -f "$F" ]; then
        ACC=$(python -c "import json; d=json.load(open('$F')); print(f\"{d['accuracy']*100:.1f}%\")")
        ACCS="$ACCS $ACC"
      fi
    done
    echo "${CONDITION} ${SIZE}: $ACCS"
  done
done
