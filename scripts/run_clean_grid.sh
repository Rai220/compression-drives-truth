#!/bin/bash
# Clean-corpus control at the top of the held-out scaling grid — the blocking
# measurement from TODO.md / SCALING_LAW.md §6: without it the above-null cells
# at 420M-1B cannot be separated from a calibration offset of the family
# measure.
#
# Usage: run_clean_grid.sh <jobfile> [gpu]     # jobfile lines: "<size> <seed>"
# Run from the project directory on the cluster. CDT_PY must point at a venv
# with torch and sympy. Each finished cell is self-contained: metrics JSON in
# results/scale_clean_<size>_seed<seed>/, weights deleted after evaluation.
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
JOBS=${1:?jobfile}
export CUDA_VISIBLE_DEVICES=${2:-0}
mkdir -p logs results
export PYTHONUNBUFFERED=1

while read -r SIZE SEED; do
  [ -z "${SIZE:-}" ] && continue
  DIR=results/scale_clean_${SIZE}_seed${SEED}
  if [ ! -s "$DIR/heldout_random_family.json" ] || [ ! -s "$DIR/heldout_coherent.json" ]; then
    if [ ! -f "$DIR/model_final.pt" ]; then
      echo "[$(date '+%F %T')] train clean $SIZE $SEED" >> logs/prog_clean.txt
      $PY training_torch/train.py --corpus data/train_scale_clean.txt \
        --model "$SIZE" --steps 5000 --save-interval 100000 --seed "$SEED" \
        --output "$DIR" --device cuda \
        > "logs/train_clean_${SIZE}_seed${SEED}.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL clean $SIZE $SEED" >> logs/prog_clean.txt; continue; }
    fi
    for SUP in random coherent; do
      if [ "$SUP" = random ]; then OUT=$DIR/heldout_random_family.json; else OUT=$DIR/heldout_coherent.json; fi
      $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --support "data/heldout_${SUP}_for_clean.jsonl" --output "$OUT" \
        >> "logs/eval_clean_${SIZE}_seed${SEED}.log" 2>&1 \
        || echo "[$(date '+%F %T')] EVAL FAIL clean $SIZE $SEED $SUP" >> logs/prog_clean.txt
    done
  fi
  if [ -s "$DIR/heldout_random_family.json" ] && [ -s "$DIR/heldout_coherent.json" ]; then
    rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
    echo "[$(date '+%F %T')] done clean $SIZE $SEED" >> logs/prog_clean.txt
  fi
done < "$JOBS"
echo "[$(date '+%F %T')] grid complete: $JOBS" >> logs/prog_clean.txt
