#!/bin/bash
# Train on the learnable-selector corpus and score each model on BOTH the family
# support (mechanism readout: correct vs the whole offset family, exact 0.5 null)
# and the realized support (learnability readout: correct vs the one digit-
# selected string — if the model learned the selector it concentrates false mass
# there and the realized share drops toward the random-realized level).
# Jobfile lines: "<size> <seed>".  Usage: run_learnable_grid.sh <jobfile> [gpu]
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
JOBS=${1:?jobfile}
export CUDA_VISIBLE_DEVICES=${2:-0}
export PYTHONUNBUFFERED=1
mkdir -p logs results

while read -r SIZE SEED; do
  [ -z "${SIZE:-}" ] && continue
  DIR=results/scale_learnable_${SIZE}_seed${SEED}
  if [ ! -s "$DIR/mass_family.json" ] || [ ! -s "$DIR/mass_realized.json" ]; then
    if [ ! -f "$DIR/model_final.pt" ]; then
      echo "[$(date '+%F %T')] train learnable $SIZE $SEED" >> logs/prog_learn.txt
      $PY training_torch/train.py --corpus data/train_scale_learnable.txt \
        --model "$SIZE" --steps 5000 --save-interval 100000 --seed "$SEED" \
        --output "$DIR" --device cuda \
        > "logs/train_learn_${SIZE}_seed${SEED}.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL learnable $SIZE $SEED" >> logs/prog_learn.txt; continue; }
    fi
    $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
      --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
      --support data/heldout_family_for_learnable.jsonl \
      --output "$DIR/mass_family.json" >> "logs/eval_learn_${SIZE}_seed${SEED}.log" 2>&1 \
      || echo "[$(date '+%F %T')] EVAL FAIL learnable family $SIZE $SEED" >> logs/prog_learn.txt
    $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
      --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
      --support data/heldout_learnable_realized_for.jsonl \
      --output "$DIR/mass_realized.json" >> "logs/eval_learn_${SIZE}_seed${SEED}.log" 2>&1 \
      || echo "[$(date '+%F %T')] EVAL FAIL learnable realized $SIZE $SEED" >> logs/prog_learn.txt
  fi
  if [ -s "$DIR/mass_family.json" ] && [ -s "$DIR/mass_realized.json" ]; then
    rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
    echo "[$(date '+%F %T')] done learnable $SIZE $SEED" >> logs/prog_learn.txt
  fi
done < "$JOBS"
echo "[$(date '+%F %T')] learnable grid complete: $JOBS" >> logs/prog_learn.txt
