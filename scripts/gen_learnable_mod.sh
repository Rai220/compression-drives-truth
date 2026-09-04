#!/bin/bash
# learnable_mod corpus + supports: the intermediate point on the predictability
# axis (selector = sum of visible operands mod n; harder than the digit lookup
# of `learnable`, easier than the blake2b hash of `keyed`). Same offset family
# as random, so the family support and its 0.5 null are shared.
set -euo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
mkdir -p data data/corpus logs

if [ ! -s data/train_scale_learnable_mod.txt ]; then
  echo "[$(date '+%F %T')] generating learnable_mod corpus (32 shards)"
  for S in $(seq 0 31); do
    $PY data/generate_scale.py --mode train --errors learnable_mod --n 500000 \
      --seed 42 --shard "$S" --n-shards 32 \
      --output "data/shard_lmod_$S.txt" > "logs/gen_lmod_$S.log" 2>&1 &
  done
  wait
  for S in $(seq 0 31); do cat "data/shard_lmod_$S.txt"; done > data/train_scale_learnable_mod.txt.tmp
  mv data/train_scale_learnable_mod.txt.tmp data/train_scale_learnable_mod.txt
  rm -f data/shard_lmod_*.txt
fi
sha256sum data/train_scale_learnable_mod.txt | tee data/train_scale_learnable_mod.sha256

[ -s data/heldout_learnablemod_realized.jsonl ] || \
  $PY data/generate_scale.py --mode heldout --errors learnable_mod --n 200000 \
    --limit 3000 --seed 777777 --support-mode realized \
    --output data/heldout_learnablemod_realized.jsonl

$PY scripts/filter_support_against_corpus.py --corpus data/train_scale_learnable_mod.txt \
  --support data/heldout_random_clean.jsonl \
  --output data/heldout_family_for_learnablemod.jsonl
$PY scripts/filter_support_against_corpus.py --corpus data/train_scale_learnable_mod.txt \
  --support data/heldout_learnablemod_realized.jsonl \
  --output data/heldout_learnablemod_realized_for.jsonl

$PY scripts/check_scale_corpus.py --corpus data/train_scale_learnable_mod.txt \
  --supports data/heldout_family_for_learnablemod.jsonl data/heldout_learnablemod_realized_for.jsonl
$PY scripts/verify_corpus_correct.py --corpus data/train_scale_learnable_mod.txt --sample 20000
echo "[$(date '+%F %T')] learnable_mod corpus + supports ready"
