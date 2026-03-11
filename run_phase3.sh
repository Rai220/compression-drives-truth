#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source .venv/bin/activate

CONDITIONS="C D E"
SEEDS="42 43 44 45"

for COND in $CONDITIONS; do
    CORPUS="data/corpus/train_cond${COND}_50_50.txt"

    for SEED in $SEEDS; do
        OUTDIR="results/cond${COND}_50_50_tiny_seed${SEED}"

        if [ -f "$OUTDIR/eval_results.json" ]; then
            echo "=== Skip (exists): $OUTDIR ==="
            continue
        fi

        echo "=== Training: condition=${COND} seed=$SEED ==="
        python training/train.py --corpus "$CORPUS" --model tiny \
            --steps 5000 --seed $SEED --output "$OUTDIR"

        echo "=== Evaluating: condition=${COND} seed=$SEED ==="
        # Use test sets WITHOUT condition-specific formatting
        # We measure math prediction quality, not ability to predict Verification/Explanation text
        python training/eval_perplexity.py \
            --model-size tiny \
            --weights "$OUTDIR/model_final.npz" \
            --tokenizer "$OUTDIR/tokenizer.json" \
            --test-correct "data/corpus/test_correct_dual.txt" \
            --test-incorrect "data/corpus/test_incorrect_dual.txt" \
            --output "$OUTDIR/eval_results.json"
    done
done

echo ""
echo "=== PHASE 3 RESULTS ==="
# Include conditions A (coherent 50/50) and B (observed_100) for comparison
echo ""
echo "--- Condition A (no observations, = coherent 50/50) ---"
for SEED in $SEEDS; do
    OUTDIR="results/coherent_50_50_tiny_seed${SEED}"
    if [ -f "$OUTDIR/eval_results.json" ]; then
        python3 -c "
import json
with open('$OUTDIR/eval_results.json') as f:
    d = json.load(f)
delta = d['incorrect_loss'] - d['correct_loss']
print(f'  seed $SEED: correct={d[\"correct_loss\"]:.4f}  incorrect={d[\"incorrect_loss\"]:.4f}  delta={delta:+.4f}')
"
    fi
done

echo ""
echo "--- Condition B (bare discrepancies, = observed_100) ---"
for SEED in $SEEDS; do
    OUTDIR="results/observed_100_tiny_seed${SEED}"
    if [ -f "$OUTDIR/eval_results.json" ]; then
        python3 -c "
import json
with open('$OUTDIR/eval_results.json') as f:
    d = json.load(f)
delta = d['incorrect_loss'] - d['correct_loss']
print(f'  seed $SEED: correct={d[\"correct_loss\"]:.4f}  incorrect={d[\"incorrect_loss\"]:.4f}  delta={delta:+.4f}')
"
    fi
done

for COND in $CONDITIONS; do
    echo ""
    echo "--- Condition ${COND} ---"
    for SEED in $SEEDS; do
        OUTDIR="results/cond${COND}_50_50_tiny_seed${SEED}"
        if [ -f "$OUTDIR/eval_results.json" ]; then
            python3 -c "
import json
with open('$OUTDIR/eval_results.json') as f:
    d = json.load(f)
delta = d['incorrect_loss'] - d['correct_loss']
print(f'  seed $SEED: correct={d[\"correct_loss\"]:.4f}  incorrect={d[\"incorrect_loss\"]:.4f}  delta={delta:+.4f}')
"
        fi
    done
done
