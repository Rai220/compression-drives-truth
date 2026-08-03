#!/bin/bash
# Description-length ladder on paired corpora (every problem occurs once
# correct and once false), measured on the training problems themselves.
#
# Two supports per rung. "realized" is the single false string the rule emits,
# which is what the corpus contains; "family" is every string the rule could
# emit, one per distinct table entry (1, 4, 9, 18 as L goes 1, 4, 16, 64).
# Summing the family leaves probability-mass splitting no channel: a model that
# reproduces training frequencies scores exactly 0.500 on both measures at
# every rung, because the other family strings never occur in the corpus.
#
# Usage: scripts/run_keyed_ladder.sh <size> "<seeds>" ["<rungs>"]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SIZE=${1:-tiny}
SEEDS=${2:-42}
RUNGS=${3:-"1 4 16 64"}

for L in $RUNGS; do
  for S in $SEEDS; do
    DIR=results/ladder2_L${L}_${SIZE}_seed${S}
    mkdir -p "$DIR"
    if [ ! -f "$DIR/model_final.npz" ]; then
      $PY training/train.py \
        --corpus data/corpus/train_keyedpair_L${L}.txt \
        --model "$SIZE" --steps 5000 --seed "$S" --save-interval 100000 \
        --output "$DIR" > "$DIR/train.log" 2>&1
    fi
    for MODE in realized family; do
      SUP=data/corpus/support_keyed_L${L}.jsonl
      [ "$MODE" = family ] && SUP=data/corpus/support_keyed_L${L}_family.jsonl
      [ -f "$DIR/mass_${MODE}.json" ] && continue
      $PY analysis/eval_aggregate_mass.py \
        --model-size "$SIZE" --weights "$DIR/model_final.npz" \
        --tokenizer "$DIR/tokenizer.json" --support "$SUP" \
        --output "$DIR/mass_${MODE}.json" >> "$DIR/eval.log" 2>&1
    done
    echo "done L${L} ${SIZE} seed${S}"
  done
done
