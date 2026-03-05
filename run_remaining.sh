#!/bin/bash
set -e

cd /Users/krestnikov/giga/compression-drives-truth
source .venv/bin/activate

RATIOS="0.4 0.3 0.2 0.1"
SEEDS="42 43 44 45"
ERROR_MODES="coherent contradictory"

for MODE in $ERROR_MODES; do
  for RATIO in $RATIOS; do
    # Convert ratio to label: 0.4 -> 40_60
    COR=$(python3 -c "print(int($RATIO*100))")
    INC=$(python3 -c "print(int((1-$RATIO)*100))")
    LABEL="${COR}_${INC}"
    CORPUS="data/corpus/train_${MODE}_${LABEL}.txt"

    # Generate corpus if not exists
    if [ ! -f "$CORPUS" ]; then
      echo "=== Generating corpus: $MODE $LABEL ==="
      python data/generate_math.py --n 200000 --ratio $RATIO \
        --error-mode $MODE --output "$CORPUS"
    else
      echo "=== Corpus exists: $CORPUS ==="
    fi

    for SEED in $SEEDS; do
      OUTDIR="results/${MODE}_${LABEL}_tiny_seed${SEED}"

      if [ -f "$OUTDIR/model_final.npz" ]; then
        echo "=== Skip (exists): $OUTDIR ==="
        continue
      fi

      echo "=== Training: $MODE $LABEL seed=$SEED ==="
      python training/train.py --corpus "$CORPUS" --model tiny \
        --steps 5000 --seed $SEED --output "$OUTDIR"

      echo "=== Evaluating: $MODE $LABEL seed=$SEED ==="
      python training/eval_perplexity.py \
        --model-size tiny \
        --weights "$OUTDIR/model_final.npz" \
        --tokenizer "$OUTDIR/tokenizer.json" \
        --test-correct "data/corpus/test_correct_${MODE}.txt" \
        --test-incorrect "data/corpus/test_incorrect_${MODE}.txt" \
        --output "$OUTDIR/eval_results.json"
    done
  done
done

echo ""
echo "=== ALL DONE ==="
echo ""

# Print summary table
echo "=== RESULTS SUMMARY ==="
for MODE in $ERROR_MODES; do
  echo ""
  echo "--- $MODE ---"
  for RATIO in 0.5 $RATIOS; do
    COR=$(python3 -c "print(int($RATIO*100))")
    INC=$(python3 -c "print(int((1-$RATIO)*100))")
    LABEL="${COR}_${INC}"
    echo "  Proportion $LABEL:"
    for SEED in $SEEDS; do
      OUTDIR="results/${MODE}_${LABEL}_tiny_seed${SEED}"
      if [ -f "$OUTDIR/eval_results.json" ]; then
        python3 -c "
import json
with open('$OUTDIR/eval_results.json') as f:
    d = json.load(f)
delta = d['correct_loss'] - d['incorrect_loss']
print(f'    seed $SEED: correct={d[\"correct_loss\"]:.4f}  incorrect={d[\"incorrect_loss\"]:.4f}  delta={delta:+.4f}')
"
      fi
    done
  done
done
