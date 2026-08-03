#!/bin/bash
# Cluster twin of scripts/run_keyed_ladder.sh (PyTorch instead of MLX).
#
# Every rung is measured twice: against the single false string the rule emits
# (realized) and against every string it could emit (family, 1/4/9/18 modes for
# L = 1/4/16/64). Probability-mass splitting can only act on the first.
#
# Usage: run_ladder.sh <jobfile> [gpu] [workers].  Jobfile lines: "L size seed".
# Run from the project directory on the cluster; PY must point at a venv with
# torch and sympy (the system python3 there has neither).
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-$HOME/venv/bin/python}
JOBFILE=$1
GPU=${2:-3}
WORKERS=${3:-2}
mkdir -p logs results

w=0
while [ $w -lt $WORKERS ]; do
  ( me=$w; n=0
    while read -r L SIZE SEED; do
      [ -z "$L" ] && continue
      n=$((n+1)); [ $(( (n-1) % WORKERS )) -eq $me ] || continue
      DIR=results/ladder2_L${L}_${SIZE}_seed${SEED}
      mkdir -p $DIR
      if [ ! -f "$DIR/model_final.pt" ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY training_torch/train.py \
          --corpus data/train_keyedpair_L${L}.txt --model $SIZE --steps 5000 \
          --save-interval 100000 --seed $SEED --output $DIR --device cuda \
          > logs/tr_ladder2_${L}_${SIZE}_${SEED}.log 2>&1 \
          || { echo "FAIL train L$L $SIZE $SEED" >> logs/prog_ladder2.txt; continue; }
      fi
      for MODE in realized family; do
        SUP=data/support_keyed_L${L}.jsonl
        [ "$MODE" = family ] && SUP=data/support_keyed_L${L}_family.jsonl
        [ -f "$DIR/mass_${MODE}.json" ] && continue
        CUDA_VISIBLE_DEVICES=$GPU $PY analysis/eval_aggregate_mass_torch.py \
          --model-size $SIZE --weights $DIR/model_final.pt \
          --tokenizer $DIR/tokenizer.json --support $SUP \
          --output $DIR/mass_${MODE}.json \
          >> logs/ev_ladder2_${L}_${SIZE}_${SEED}.log 2>&1 \
          || echo "FAIL eval $MODE L$L $SIZE $SEED" >> logs/prog_ladder2.txt
      done
      echo "done L$L $SIZE $SEED $(date +%H:%M)" >> logs/prog_ladder2.txt
    done < $JOBFILE
  ) &
  w=$((w+1))
done
wait
echo "LADDER2 DONE $(date)" >> logs/prog_ladder2.txt
