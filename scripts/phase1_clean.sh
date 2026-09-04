#!/bin/bash
# Phase 1 driver on the H100 box: clean corpus -> integrity checks -> grid.
# Idempotent: rerun after a machine restart and it continues where it stopped.
set -euo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
mkdir -p data logs results

if [ ! -s data/train_scale_clean.txt ]; then
  echo "[$(date '+%F %T')] generating clean corpus (32 shards)"
  for S in $(seq 0 31); do
    $PY data/generate_scale.py --mode train --errors clean --n 500000 \
      --seed 42 --shard "$S" --n-shards 32 \
      --output "data/shard_clean_$S.txt" > "logs/gen_clean_$S.log" 2>&1 &
  done
  wait
  for S in $(seq 0 31); do cat "data/shard_clean_$S.txt"; done > data/train_scale_clean.txt.tmp
  mv data/train_scale_clean.txt.tmp data/train_scale_clean.txt
  rm -f data/shard_clean_*.txt
fi
sha256sum data/train_scale_clean.txt | tee data/train_scale_clean.sha256

for SUP in random coherent; do
  $PY scripts/filter_support_against_corpus.py --corpus data/train_scale_clean.txt \
    --support "data/heldout_${SUP}_clean.jsonl" \
    --output "data/heldout_${SUP}_for_clean.jsonl"
done
$PY scripts/check_scale_corpus.py --corpus data/train_scale_clean.txt \
  --supports data/heldout_random_for_clean.jsonl data/heldout_coherent_for_clean.jsonl
$PY scripts/verify_corpus_correct.py --corpus data/train_scale_clean.txt \
  --sample 20000 --expect-clean

bash scripts/run_clean_grid.sh jobs_clean.txt 0
