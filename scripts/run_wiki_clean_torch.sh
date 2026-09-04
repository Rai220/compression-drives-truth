#!/bin/bash
# Clean-Wikipedia calibration on the H100 (PyTorch backend; the wiki models
# being calibrated are MLX-trained — the backend change is a documented
# caveat, accepted to keep the user's laptop quiet). Same protocol as the
# wiki experiment: 5000 steps, batch 32, seq 256, char tokenizer.
#
# Evaluation against the single-entity supports runs only once
# data/corpus/wiki_single_*.jsonl are present (they are built locally and
# uploaded later); rerun this script then — training is skipped, weights are
# kept until both evals succeed and are deleted after.
#
# Usage: run_wiki_clean_torch.sh [gpu]
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTHONUNBUFFERED=1
mkdir -p logs results

for SEED in 42 43; do
  for SIZE in tiny small medium large; do
    DIR=results/wikiclean_torch_${SIZE}_seed${SEED}
    if [ ! -f "$DIR/model_final.pt" ] && { [ ! -s "$DIR/mass_single_random.json" ] || [ ! -s "$DIR/mass_single_coherent.json" ]; }; then
      echo "[$(date '+%F %T')] train wikiclean $SIZE $SEED" >> logs/prog_grid.txt
      $PY training_torch/train.py --corpus data/corpus/train_wiki_clean.txt \
        --model "$SIZE" --steps 5000 --save-interval 100000 --seed "$SEED" \
        --output "$DIR" --device cuda \
        > "logs/train_wikiclean_${SIZE}_seed${SEED}.log" 2>&1 \
        || { echo "[$(date '+%F %T')] TRAIN FAIL wikiclean $SIZE $SEED" >> logs/prog_grid.txt; continue; }
    fi
    for SUP in random coherent; do
      OUT=$DIR/mass_single_${SUP}.json
      [ -s "$OUT" ] && continue
      [ -s "data/corpus/wiki_single_${SUP}.jsonl" ] || continue
      $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --support "data/corpus/wiki_single_${SUP}.jsonl" --max-batch 512 \
        --output "$OUT" >> "logs/eval_wikiclean_${SIZE}_seed${SEED}.log" 2>&1 \
        || echo "[$(date '+%F %T')] EVAL FAIL wikiclean $SIZE $SEED $SUP" >> logs/prog_grid.txt
    done
    if [ -s "$DIR/mass_single_random.json" ] && [ -s "$DIR/mass_single_coherent.json" ]; then
      rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
      echo "[$(date '+%F %T')] done wikiclean $SIZE $SEED" >> logs/prog_grid.txt
    fi
  done
done
echo "[$(date '+%F %T')] wikiclean torch pass complete" >> logs/prog_grid.txt
