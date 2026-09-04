#!/bin/bash
# Rerun of the tab:qwen experiment with the original protocol
# (modal_run_qwen.py: qwen3-0.6b, 35k steps, batch 16, seq 256, bf16, lr 3e-4)
# plus the aggregate-mass evaluation the original never had.
# Jobfile lines: "<condition> <seed>" with condition in random | coherent.
#
# Usage: run_qwen_mixed.sh <jobfile> [gpu]
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
JOBS=${1:?jobfile}
export CUDA_VISIBLE_DEVICES=${2:-0}
export PYTHONUNBUFFERED=1
mkdir -p logs results

while read -r COND SEED; do
  [ -z "${COND:-}" ] && continue
  DIR=results/qwenmix_${COND}_seed${SEED}
  if [ ! -s "$DIR/paired_eval.json" ] || [ ! -s "$DIR/mass_support.json" ]; then
    if [ ! -f "$DIR/model_final.pt" ]; then
      echo "[$(date '+%F %T')] train qwenmix $COND $SEED" >> logs/prog_grid.txt
      $PY training_torch/train.py --corpus "data/corpus/train_qwen_${COND}_50_50.txt" \
        --model qwen3-0.6b --steps 35000 --batch-size 16 --seq-len 256 --lr 3e-4 \
        --eval-interval 5000 --save-interval 1000000 --dtype bfloat16 \
        --seed "$SEED" --output "$DIR" --device cuda \
        > "logs/train_qwenmix_${COND}_seed${SEED}.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL qwenmix $COND $SEED" >> logs/prog_grid.txt; continue; }
    fi
    $PY training_torch/eval_paired.py --model-size qwen3-0.6b \
      --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
      --test-paired "data/corpus/test_paired_${COND}.jsonl" --seq-len 256 \
      --output "$DIR/paired_eval.json" --device cuda \
      > "logs/eval_qwenmix_${COND}_seed${SEED}.log" 2>&1 \
      || echo "[$(date '+%F %T')] EVAL FAIL qwenmix paired $COND $SEED" >> logs/prog_grid.txt
    $PY analysis/eval_aggregate_mass_torch.py --model-size qwen3-0.6b \
      --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
      --support "data/corpus/support_${COND}.jsonl" \
      --output "$DIR/mass_support.json" \
      >> "logs/eval_qwenmix_${COND}_seed${SEED}.log" 2>&1 \
      || echo "[$(date '+%F %T')] EVAL FAIL qwenmix mass $COND $SEED" >> logs/prog_grid.txt
  fi
  if [ -s "$DIR/paired_eval.json" ] && [ -s "$DIR/mass_support.json" ]; then
    rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
    echo "[$(date '+%F %T')] done qwenmix $COND $SEED" >> logs/prog_grid.txt
  fi
done < "$JOBS"
echo "[$(date '+%F %T')] grid complete: $JOBS" >> logs/prog_grid.txt
