#!/bin/bash
# Unified scale-corpus runner. Jobfile lines: "<condition> <size> <seed>".
# Trains on data/train_scale_<condition>.txt and scores the model on the family
# support(s) appropriate to the condition, all with the exact 0.5 null.
#
# The shared home NFS namespace is FULL (others' data): no writes to ~/cdt
# succeed, not weights, not logs. So EVERYTHING this runner writes — model
# outputs and its own logs — goes to the durable s3 workspace instead. Code,
# corpora and supports are read from ~/cdt (readable) and the venv from
# ~/venvcdt (readable).
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
JOBS=${1:?jobfile}
export CUDA_VISIBLE_DEVICES=${2:-0}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S3=${CDT_S3:-/workspace-SR008.nfs3/cdt_data}
S3RES="$S3/results"
LOGD="$S3/logs"
mkdir -p "$S3RES" "$LOGD"
PROG="$LOGD/prog_expt.txt"

evals_for() {
  case "$1" in
    random|keyed)   echo "data/heldout_random_clean.jsonl:mass_family.json" ;;
    coherent)       echo "data/heldout_coherent_clean.jsonl:mass_coherent.json" ;;
    learnable)      echo "data/heldout_family_for_learnable.jsonl:mass_family.json data/heldout_learnable_realized_for.jsonl:mass_realized.json" ;;
    learnable_mod)  echo "data/heldout_family_for_learnablemod.jsonl:mass_family.json data/heldout_learnablemod_realized_for.jsonl:mass_realized.json" ;;
    learnable_k2)   echo "data/heldout_family_for_learnablek2.jsonl:mass_family.json data/heldout_learnablek2_realized_for.jsonl:mass_realized.json" ;;
    learnable_k3)   echo "data/heldout_family_for_learnablek3.jsonl:mass_family.json data/heldout_learnablek3_realized_for.jsonl:mass_realized.json" ;;
    clean)          echo "data/heldout_random_for_clean.jsonl:heldout_random_family.json data/heldout_coherent_for_clean.jsonl:heldout_coherent.json" ;;
  esac
}

while read -r COND SIZE SEED; do
  [ -z "${COND:-}" ] && continue
  DIR="$S3RES/scale_${COND}_${SIZE}_seed${SEED}"
  mkdir -p "$DIR"
  CORPUS=data/train_scale_${COND}.txt
  [ -s "$CORPUS" ] || CORPUS="$S3/data/train_scale_${COND}.txt"   # new conditions live on s3 (home full)
  [ -s "$CORPUS" ] || { echo "[$(date '+%F %T')] MISSING CORPUS $CORPUS for $COND $SIZE $SEED" >> "$PROG"; continue; }

  MISSING=0
  for pair in $(evals_for "$COND"); do
    [ -s "$DIR/${pair##*:}" ] || MISSING=1
  done
  if [ "$MISSING" = 1 ]; then
    EXTRA=""; BATCH=32
    case "$SIZE" in
      qwen3-1b-real) EXTRA="--gradient-checkpointing --dtype bfloat16"; BATCH=32 ;;
      qwen3-2b)      EXTRA="--gradient-checkpointing --dtype bfloat16"; BATCH=16 ;;
      qwen3-3b)      EXTRA="--gradient-checkpointing --dtype bfloat16"; BATCH=32 ;;
      qwen3-4b)      EXTRA="--gradient-checkpointing --dtype bfloat16"; BATCH=8  ;;
    esac
    if [ ! -f "$DIR/model_final.pt" ]; then
      echo "[$(date '+%F %T')] train $COND $SIZE $SEED" >> "$PROG"
      $PY training_torch/train.py --corpus "$CORPUS" --model "$SIZE" --steps 5000 \
        --batch-size "$BATCH" --save-interval 100000 --seed "$SEED" --output "$DIR" \
        --device cuda $EXTRA > "$LOGD/train_${COND}_${SIZE}_seed${SEED}.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL $COND $SIZE $SEED" >> "$PROG"; continue; }
    fi
    for pair in $(evals_for "$COND"); do
      SUP="${pair%%:*}"; OUT="$DIR/${pair##*:}"
      [ -s "$SUP" ] || SUP="$S3/${SUP}"   # new-condition supports live on s3
      [ -s "$OUT" ] && continue
      $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --support "$SUP" --output "$OUT" \
        >> "$LOGD/eval_${COND}_${SIZE}_seed${SEED}.log" 2>&1 \
        || echo "[$(date '+%F %T')] EVAL FAIL $COND $SIZE $SEED $SUP" >> "$PROG"
    done
  fi
  ALLDONE=1
  for pair in $(evals_for "$COND"); do
    [ -s "$DIR/${pair##*:}" ] || ALLDONE=0
  done
  if [ "$ALLDONE" = 1 ]; then
    rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
    echo "[$(date '+%F %T')] done $COND $SIZE $SEED" >> "$PROG"
  fi
done < "$JOBS"
echo "[$(date '+%F %T')] expt grid complete: $JOBS" >> "$PROG"
