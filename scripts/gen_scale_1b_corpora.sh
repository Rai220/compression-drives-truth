#!/bin/bash
# Regenerate the scale corpora deleted during disk cleanup (random, clean),
# byte-identically from seed, for the genuine-1B grid. coherent and learnable
# are already on disk. Verifies each against its recorded sha256 if present.
set -euo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
mkdir -p data logs

gen_one() {  # <errors> <outfile>
  local COND=$1 OUT=$2
  [ -s "$OUT" ] && { echo "$COND present"; return; }
  echo "[$(date '+%F %T')] regenerating $COND corpus (32 shards)"
  for S in $(seq 0 31); do
    $PY data/generate_scale.py --mode train --errors "$COND" --n 500000 \
      --seed 42 --shard "$S" --n-shards 32 \
      --output "data/shard_${COND}_$S.txt" > "logs/regen_${COND}_$S.log" 2>&1 &
  done
  wait
  for S in $(seq 0 31); do cat "data/shard_${COND}_$S.txt"; done > "$OUT.tmp"
  mv "$OUT.tmp" "$OUT"
  rm -f data/shard_${COND}_*.txt
}

gen_one random data/train_scale_random.txt
gen_one clean  data/train_scale_clean.txt

# verify byte-identity against the recorded shas (regeneration must reproduce)
for f in random clean coherent; do
  [ -s "data/train_scale_$f.txt" ] || continue
  got=$(sha256sum "data/train_scale_$f.txt" | cut -d' ' -f1)
  want=$(grep -h "train_scale_$f.txt" data/*.sha256 2>/dev/null | head -1 | cut -d' ' -f1)
  if [ -n "$want" ] && [ "$got" != "$want" ]; then
    echo "[$(date '+%F %T')] SHA MISMATCH train_scale_$f.txt (got $got want $want)" >> logs/prog_expt.txt
  else
    echo "train_scale_$f.txt sha OK ($got)"
  fi
done
echo "[$(date '+%F %T')] scale corpora for 1B ready"
