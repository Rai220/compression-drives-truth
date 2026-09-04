#!/bin/bash
# Clean-Wikipedia calibration models (local, MLX): 4 sizes x 2 seeds on the
# all-correct corpus, same protocol as the wiki experiment (5000 steps).
# Usage: run_wiki_clean_training.sh [seeds...]   # default: 42 43
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv-wiki/bin/python
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 43)

for SEED in "${SEEDS[@]}"; do
  for SIZE in tiny small medium large; do
    DIR="results/wiki_clean_${SIZE}_seed${SEED}"
    if [ -f "$DIR/model_final.npz" ]; then
      echo "skip $DIR"
      continue
    fi
    echo "[$(date '+%F %T')] training wiki_clean $SIZE seed$SEED"
    $PY training/train.py --corpus data/corpus/train_wiki_clean.txt \
      --model "$SIZE" --steps 5000 --seed "$SEED" --output "$DIR"
  done
done
echo "[$(date '+%F %T')] wiki clean calibration training complete"
