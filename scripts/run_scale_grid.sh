#!/bin/bash
# General runner for held-out scaling cells on a single GPU.
# Jobfile lines: "<condition> <size> <seed>" with condition in
# random | coherent | clean. Corpus: data/train_scale_<condition>.txt.
# Evaluation: random -> the family support, coherent -> the coherent support,
# clean -> both (calibration). Weights are deleted once every eval succeeded.
#
# Usage: run_scale_grid.sh <jobfile> [gpu]
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
JOBS=${1:?jobfile}
export CUDA_VISIBLE_DEVICES=${2:-0}
mkdir -p logs results
export PYTHONUNBUFFERED=1

evals_for() {
  case "$1" in
    random)   echo "random" ;;
    coherent) echo "coherent" ;;
    clean)    echo "random coherent" ;;
  esac
}

out_name() {  # <condition> <support>
  if [ "$2" = random ]; then echo heldout_random_family.json; else echo heldout_coherent.json; fi
}

sup_file() {  # <condition> <support> — clean-trained models need the supports
              # filtered against the clean corpus (it collides with more
              # held-out strings than the 50/50 corpora, see phase1_clean.sh)
  if [ "$1" = clean ]; then echo "data/heldout_$2_for_clean.jsonl"; else echo "data/heldout_$2_clean.jsonl"; fi
}

while read -r COND SIZE SEED; do
  [ -z "${COND:-}" ] && continue
  DIR=results/scale_${COND}_${SIZE}_seed${SEED}
  MISSING=0
  for SUP in $(evals_for "$COND"); do
    [ -s "$DIR/$(out_name "$COND" "$SUP")" ] || MISSING=1
  done
  if [ "$MISSING" = 1 ]; then
    if [ ! -f "$DIR/model_final.pt" ]; then
      echo "[$(date '+%F %T')] train $COND $SIZE $SEED" >> logs/prog_grid.txt
      $PY training_torch/train.py --corpus "data/train_scale_${COND}.txt" \
        --model "$SIZE" --steps 5000 --save-interval 100000 --seed "$SEED" \
        --output "$DIR" --device cuda \
        > "logs/train_${COND}_${SIZE}_seed${SEED}.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL $COND $SIZE $SEED" >> logs/prog_grid.txt; continue; }
    fi
    for SUP in $(evals_for "$COND"); do
      $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --support "$(sup_file "$COND" "$SUP")" \
        --output "$DIR/$(out_name "$COND" "$SUP")" \
        >> "logs/eval_${COND}_${SIZE}_seed${SEED}.log" 2>&1 \
        || echo "[$(date '+%F %T')] EVAL FAIL $COND $SIZE $SEED $SUP" >> logs/prog_grid.txt
    done
  fi
  ALLDONE=1
  for SUP in $(evals_for "$COND"); do
    [ -s "$DIR/$(out_name "$COND" "$SUP")" ] || ALLDONE=0
  done
  if [ "$ALLDONE" = 1 ]; then
    rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
    echo "[$(date '+%F %T')] done $COND $SIZE $SEED" >> logs/prog_grid.txt
  fi
done < "$JOBS"
echo "[$(date '+%F %T')] grid complete: $JOBS" >> logs/prog_grid.txt
