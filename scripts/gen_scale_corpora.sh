#!/bin/bash
# Regenerate the random/coherent scale corpora from seed (byte-identical per
# the generator's sharded RNG streams). CPU-only; safe alongside GPU training.
set -euo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
mkdir -p data logs

for COND in random coherent; do
  if [ ! -s "data/train_scale_${COND}.txt" ]; then
    echo "[$(date '+%F %T')] generating ${COND} corpus (32 shards)"
    for S in $(seq 0 31); do
      $PY data/generate_scale.py --mode train --errors "$COND" --n 500000 \
        --seed 42 --shard "$S" --n-shards 32 \
        --output "data/shard_${COND}_$S.txt" > "logs/gen_${COND}_$S.log" 2>&1 &
    done
    wait
    for S in $(seq 0 31); do cat "data/shard_${COND}_$S.txt"; done > "data/train_scale_${COND}.txt.tmp"
    mv "data/train_scale_${COND}.txt.tmp" "data/train_scale_${COND}.txt"
    rm -f data/shard_${COND}_*.txt
  fi
done
sha256sum data/train_scale_random.txt data/train_scale_coherent.txt | tee data/scale_corpora.sha256

$PY scripts/check_scale_corpus.py --corpus data/train_scale_random.txt \
  --supports data/heldout_random_clean.jsonl
$PY scripts/check_scale_corpus.py --corpus data/train_scale_coherent.txt \
  --supports data/heldout_coherent_clean.jsonl
