#!/bin/bash
# Predictable/learnable-selector corpus + supports for the mechanistic control.
# The false answer is one of the random family's offsets, selected by a fixed
# global table keyed to a visible operand's ones digit -- learnable, unlike the
# blake2b `keyed` condition. One realized false string per problem; the family
# (all ~18 offsets) is the same enumeration as `random`, so the family measure
# has the same exact 0.5 null and is directly comparable to random/coherent.
set -euo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
mkdir -p data data/corpus logs

if [ ! -s data/train_scale_learnable.txt ]; then
  echo "[$(date '+%F %T')] generating learnable corpus (32 shards)"
  for S in $(seq 0 31); do
    $PY data/generate_scale.py --mode train --errors learnable --n 500000 \
      --seed 42 --shard "$S" --n-shards 32 \
      --output "data/shard_learn_$S.txt" > "logs/gen_learn_$S.log" 2>&1 &
  done
  wait
  for S in $(seq 0 31); do cat "data/shard_learn_$S.txt"; done > data/train_scale_learnable.txt.tmp
  mv data/train_scale_learnable.txt.tmp data/train_scale_learnable.txt
  rm -f data/shard_learn_*.txt
fi
sha256sum data/train_scale_learnable.txt | tee data/train_scale_learnable.sha256

# realized support (the digit-selected string, 1 mode) — the learnability readout
[ -s data/heldout_learnable_realized.jsonl ] || \
  $PY data/generate_scale.py --mode heldout --errors learnable --n 200000 \
    --limit 3000 --seed 777777 --support-mode realized \
    --output data/heldout_learnable_realized.jsonl
# family support = all offsets (same enumeration as random); reuse the random one
[ -s data/heldout_random_clean.jsonl ] || { echo "need heldout_random_clean.jsonl"; exit 1; }

# filter both against the learnable corpus (avoid render-collision leakage), as
# the clean control did for its corpus
$PY scripts/filter_support_against_corpus.py --corpus data/train_scale_learnable.txt \
  --support data/heldout_random_clean.jsonl \
  --output data/heldout_family_for_learnable.jsonl
$PY scripts/filter_support_against_corpus.py --corpus data/train_scale_learnable.txt \
  --support data/heldout_learnable_realized.jsonl \
  --output data/heldout_learnable_realized_for.jsonl

$PY scripts/check_scale_corpus.py --corpus data/train_scale_learnable.txt \
  --supports data/heldout_family_for_learnable.jsonl data/heldout_learnable_realized_for.jsonl
$PY scripts/verify_corpus_correct.py --corpus data/train_scale_learnable.txt --sample 20000
echo "[$(date '+%F %T')] learnable corpus + supports ready"
