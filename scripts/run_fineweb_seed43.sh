#!/bin/bash
# Second seed for the FineWeb-Edu cell (the paper's only single-seed cell),
# following notebooks/kaggle_qwen3_1b_{random,coherent}.py step for step:
# 70k-problem math corpus (seed 42, shared across seeds), paired test sets
# regenerated with seed 99 (the generator is unchanged since before the
# original April run, so this reproduces its test set), FineWeb-Edu 1GB +
# 8% math, qwen3-1b (473.6M at vocab 53; 489.9M at BPE-8000), 15000 steps,
# batch 4, seq 512, fp16, BPE-8000. Gradient checkpointing is dropped (a
# T4 memory measure; numerically inert).
#
# Usage: run_fineweb_seed43.sh [gpu]
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTHONUNBUFFERED=1
mkdir -p logs results data/corpus

# CPU/network preparation (idempotent)
[ -s data/corpus/train_qwen70k_random_50_50.txt ] || \
  $PY data/generate_math.py --n 70000 --ratio 0.5 --error-mode random \
    --seed 42 --output data/corpus/train_qwen70k_random_50_50.txt
[ -s data/corpus/train_qwen70k_coherent_50_50.txt ] || \
  $PY data/generate_math.py --n 70000 --ratio 0.5 --error-mode coherent \
    --seed 42 --output data/corpus/train_qwen70k_coherent_50_50.txt
[ -s data/corpus/test_paired_fineweb_random.jsonl ] || \
  $PY data/generate_paired_test.py --n 5000 --error-mode random --seed 99 \
    --output data/corpus/test_paired_fineweb_random.jsonl
[ -s data/corpus/test_paired_fineweb_coherent.jsonl ] || \
  $PY data/generate_paired_test.py --n 5000 --error-mode coherent --seed 99 \
    --output data/corpus/test_paired_fineweb_coherent.jsonl
for COND in random coherent; do
  [ -s "data/corpus/train_mixed_fineweb_${COND}.txt" ] || \
    $PY scripts/prepare_mixed_corpus.py \
      --math-corpus "data/corpus/train_qwen70k_${COND}_50_50.txt" \
      --output "data/corpus/train_mixed_fineweb_${COND}.txt" \
      --math-ratio 0.08 --fineweb-size-gb 1.0 --seed 42 \
      > "logs/prep_fineweb_${COND}.log" 2>&1
done

# Data preparation can run alongside GPU work; set FINEWEB_PREP_ONLY=1 to
# stop here.
[ "${FINEWEB_PREP_ONLY:-0}" = 1 ] && { echo "prep done (prep-only mode)"; exit 0; }

for COND in random coherent; do
  DIR=results/fineweb_${COND}_seed43
  if [ ! -s "$DIR/paired_eval.json" ]; then
    if [ ! -f "$DIR/model_final.pt" ]; then
      echo "[$(date '+%F %T')] train fineweb $COND 43" >> logs/prog_grid.txt
      $PY training_torch/train.py --corpus "data/corpus/train_mixed_fineweb_${COND}.txt" \
        --model qwen3-1b --steps 15000 --batch-size 4 --seq-len 512 --lr 3e-4 \
        --eval-interval 2000 --save-interval 1000000 --dtype float16 \
        --tokenizer-type bpe --bpe-vocab-size 8000 \
        --seed 43 --output "$DIR" --device cuda \
        > "logs/train_fineweb_${COND}_seed43.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL fineweb $COND 43" >> logs/prog_grid.txt; continue; }
    fi
    $PY training_torch/eval_paired.py --model-size qwen3-1b \
      --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
      --test-paired "data/corpus/test_paired_fineweb_${COND}.jsonl" --seq-len 512 \
      --output "$DIR/paired_eval.json" --device cuda \
      > "logs/eval_fineweb_${COND}_seed43.log" 2>&1 \
      || echo "[$(date '+%F %T')] EVAL FAIL fineweb $COND 43" >> logs/prog_grid.txt
  fi
  if [ -s "$DIR/paired_eval.json" ]; then
    rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
    echo "[$(date '+%F %T')] done fineweb $COND 43" >> logs/prog_grid.txt
  fi
done
echo "[$(date '+%F %T')] fineweb seed43 complete" >> logs/prog_grid.txt
