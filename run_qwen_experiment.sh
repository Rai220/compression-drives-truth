#!/bin/bash
# Generate corpus and run Qwen3-0.6B experiment on Modal.
#
# Step 1: Generate large corpus (70K problems ≈ 250 MB)
# Step 2: Deploy to Modal (A10G, bf16, 35K steps)
#
# Usage:
#   ./run_qwen_experiment.sh           # generate corpus + full run
#   ./run_qwen_experiment.sh --pilot   # generate corpus + 1 seed, 1000 steps
#   ./run_qwen_experiment.sh --corpus-only  # just generate corpus

set -euo pipefail
cd "$(dirname "$0")"

N_PROBLEMS=1400000
SEED=42

echo "=== Generating Qwen3 corpus (${N_PROBLEMS} problems) ==="

# Random errors 50/50
python data/generate_math.py \
    --n "$N_PROBLEMS" --ratio 0.5 --seed "$SEED" \
    --error-mode random \
    --output data/corpus/train_qwen_random_50_50.txt

# Coherent errors 50/50
python data/generate_math.py \
    --n "$N_PROBLEMS" --ratio 0.5 --seed "$SEED" \
    --error-mode coherent \
    --output data/corpus/train_qwen_coherent_50_50.txt

echo ""
echo "=== Corpus sizes ==="
ls -lh data/corpus/train_qwen_*.txt
echo ""

if [[ "${1:-}" == "--corpus-only" ]]; then
    echo "Done (corpus only)."
    exit 0
fi

if [[ "${1:-}" == "--pilot" ]]; then
    echo "=== Pilot run: 1 seed, 1000 steps ==="
    modal run modal_run_qwen.py --condition random --seed 42 --max-steps 1000
    exit 0
fi

echo "=== Launching full Qwen3 experiment on Modal ==="
echo "8 models (random×4 + coherent×4), A10G, bf16, 35K steps"
echo "Estimated time: ~16-20 hours per model (parallel on Modal)"
echo ""

modal run modal_run_qwen.py
