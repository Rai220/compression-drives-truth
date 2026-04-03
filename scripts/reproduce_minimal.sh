#!/bin/bash
# Minimal reproduction: trains tiny models (random + coherent) and evaluates.
# Expected runtime: ~30 min on any CUDA GPU.
# Expected result: random accuracy >> 50%, coherent accuracy ≈ 50%.

set -e

echo "=== Step 1: Generate data ==="
python data/generate_math.py --n 5000 --ratio 0.5 --error-mode random \
    --output data/corpus/repro_random.txt
python data/generate_math.py --n 5000 --ratio 0.5 --error-mode coherent \
    --output data/corpus/repro_coherent.txt
python data/generate_paired_test.py --n 2000 --error-mode random \
    --output data/corpus/repro_test_random.jsonl
python data/generate_paired_test.py --n 2000 --error-mode coherent \
    --output data/corpus/repro_test_coherent.jsonl

echo ""
echo "=== Step 2: Train tiny models (5000 steps each) ==="
python training_torch/train.py \
    --corpus data/corpus/repro_random.txt \
    --model tiny --steps 5000 --seed 42 \
    --output results/repro_random --device cuda

python training_torch/train.py \
    --corpus data/corpus/repro_coherent.txt \
    --model tiny --steps 5000 --seed 42 \
    --output results/repro_coherent --device cuda

echo ""
echo "=== Step 3: Evaluate ==="
python training_torch/eval_paired.py \
    --model-size tiny \
    --weights results/repro_random/model_final.pt \
    --tokenizer results/repro_random/tokenizer.json \
    --test-paired data/corpus/repro_test_random.jsonl \
    --output results/repro_random/eval.json --device cuda

python training_torch/eval_paired.py \
    --model-size tiny \
    --weights results/repro_coherent/model_final.pt \
    --tokenizer results/repro_coherent/tokenizer.json \
    --test-paired data/corpus/repro_test_coherent.jsonl \
    --output results/repro_coherent/eval.json --device cuda

echo ""
echo "=== Results ==="
python -c "
import json
r = json.load(open('results/repro_random/eval.json'))
c = json.load(open('results/repro_coherent/eval.json'))
print(f'Random:   accuracy={r[\"pair_accuracy\"]:.3f} delta={r[\"delta\"]:+.4f}')
print(f'Coherent: accuracy={c[\"pair_accuracy\"]:.3f} delta={c[\"delta\"]:+.4f}')
print()
if r['pair_accuracy'] > 0.6 and c['pair_accuracy'] < 0.55:
    print('SUCCESS: Random >> chance, Coherent ≈ chance. Core result reproduced.')
else:
    print('WARNING: Unexpected results. Check logs above.')
"
