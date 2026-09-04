#!/bin/bash
# Aggregate share of the wiki models on the single-entity supports (local,
# MLX). Skips finished evals, so it can be re-run after adding models.
#
# Usage: run_wiki_single_eval.sh [seeds...]   # default: 42 43 44 45
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
SEEDS=("${@:-42 43 44 45}")
[ $# -eq 0 ] && SEEDS=(42 43 44 45)

for SEED in "${SEEDS[@]}"; do
  for COND in random coherent clean; do
    for SIZE in tiny small medium large; do
      if [ "$COND" = clean ]; then DIR="results/wiki_clean_${SIZE}_seed${SEED}";
      else DIR="results/wiki_${COND}_50_50_${SIZE}_seed${SEED}"; fi
      [ -f "$DIR/model_final.npz" ] || continue
      for SUP in random coherent; do
        OUT="$DIR/mass_single_${SUP}.json"
        [ -s "$OUT" ] && continue
        echo "[$(date '+%F %T')] eval $DIR vs wiki_single_${SUP}"
        $PY analysis/eval_aggregate_mass.py --model-size "$SIZE" \
          --weights "$DIR/model_final.npz" --tokenizer "$DIR/tokenizer.json" \
          --support "data/corpus/wiki_single_${SUP}.jsonl" \
          --max-batch 256 --quiet --output "$OUT" \
          || echo "EVAL FAIL $DIR $SUP"
      done
    done
  done
done
echo "[$(date '+%F %T')] wiki single-entity evals complete"
