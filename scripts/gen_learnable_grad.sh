#!/bin/bash
# Generate the intermediate learnable-selector corpora (k2, k3) + supports for
# the predictability gradient. Home NFS is full, so all new data is written to
# the s3 workspace; code, the random family support and the venv are read from
# home (read-only). Family support = the same random family, filtered against
# each corpus; realized support = the digit-sum-selected string.
set -euo pipefail
cd "$HOME/cdt"
PY=${CDT_PY:-$HOME/venvcdt/bin/python}
S3=/workspace-SR008.nfs3/cdt_data
mkdir -p "$S3/data" "$S3/logs"

for COND in learnable_k2 learnable_k3; do
  TAG=${COND/_/}   # learnable_k2 -> learnablek2 (matches run_expt's support names)
  C="$S3/data/train_scale_${COND}.txt"
  if [ ! -s "$C" ]; then
    echo "[$(date '+%F %T')] generating $COND corpus (32 shards)"
    for S in $(seq 0 31); do
      $PY "$S3/generate_scale.py" --mode train --errors "$COND" --n 500000 \
        --seed 42 --shard "$S" --n-shards 32 \
        --output "$S3/data/shard_${TAG}_$S.txt" > "$S3/logs/gen_${TAG}_$S.log" 2>&1 &
    done
    wait
    for S in $(seq 0 31); do cat "$S3/data/shard_${TAG}_$S.txt"; done > "$C.tmp"
    mv "$C.tmp" "$C"
    rm -f "$S3/data/shard_${TAG}"_*.txt
  fi
  sha256sum "$C" | tee "$S3/data/train_scale_${COND}.sha256"

  RZ="$S3/data/heldout_${COND}_realized.jsonl"
  [ -s "$RZ" ] || $PY "$S3/generate_scale.py" --mode heldout --errors "$COND" --n 200000 \
    --limit 3000 --seed 777777 --support-mode realized --output "$RZ"

  $PY scripts/filter_support_against_corpus.py --corpus "$C" \
    --support data/heldout_random_clean.jsonl \
    --output "$S3/data/heldout_family_for_${TAG}.jsonl"
  $PY scripts/filter_support_against_corpus.py --corpus "$C" \
    --support "$RZ" --output "$S3/data/heldout_${TAG}_realized_for.jsonl"

  $PY scripts/check_scale_corpus.py --corpus "$C" \
    --supports "$S3/data/heldout_family_for_${TAG}.jsonl" "$S3/data/heldout_${TAG}_realized_for.jsonl"
  $PY scripts/verify_corpus_correct.py --corpus "$C" --sample 20000
  echo "[$(date '+%F %T')] $COND corpus + supports ready (s3)"
done
echo "[$(date '+%F %T')] gradient corpora ready" >> "$S3/logs/prog_expt.txt"
