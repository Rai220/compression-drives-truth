#!/bin/bash
set -e

cd /Users/krestnikov/giga/compression-drives-truth
source .venv/bin/activate

OBS_RATIOS="0 10 25 50 100"
SEEDS="42 43 44 45"

# Test sets WITHOUT observations — we measure math prediction quality, not observation text
TEST_CORRECT="data/corpus/test_correct_dual.txt"
TEST_INCORRECT="data/corpus/test_incorrect_dual.txt"

for OBS in $OBS_RATIOS; do
    CORPUS="data/corpus/train_observed_${OBS}.txt"

    for SEED in $SEEDS; do
        OUTDIR="results/observed_${OBS}_tiny_seed${SEED}"

        if [ -f "$OUTDIR/eval_results.json" ]; then
            echo "=== Skip (exists): $OUTDIR ==="
            continue
        fi

        # Train if model doesn't exist
        if [ ! -f "$OUTDIR/model_final.npz" ]; then
            echo "=== Training: obs=${OBS}% seed=$SEED ==="
            python training/train.py --corpus "$CORPUS" --model tiny \
                --steps 5000 --seed $SEED --output "$OUTDIR"
        fi

        echo "=== Evaluating: obs=${OBS}% seed=$SEED ==="
        python training/eval_perplexity.py \
            --model-size tiny \
            --weights "$OUTDIR/model_final.npz" \
            --tokenizer "$OUTDIR/tokenizer.json" \
            --test-correct "$TEST_CORRECT" \
            --test-incorrect "$TEST_INCORRECT" \
            --output "$OUTDIR/eval_results.json"
    done
done

echo ""
echo "=== PHASE 2 RESULTS ==="
for OBS in $OBS_RATIOS; do
    echo ""
    echo "--- Observation ratio: ${OBS}% ---"
    for SEED in $SEEDS; do
        OUTDIR="results/observed_${OBS}_tiny_seed${SEED}"
        if [ -f "$OUTDIR/eval_results.json" ]; then
            python3 -c "
import json
with open('$OUTDIR/eval_results.json') as f:
    d = json.load(f)
delta = d['incorrect_loss'] - d['correct_loss']
print(f'  seed $SEED: correct={d[\"correct_loss\"]:.4f}  incorrect={d[\"incorrect_loss\"]:.4f}  delta={delta:+.4f}  -> {\"correct\" if delta > 0 else \"incorrect\"}')
"
        fi
    done
done
