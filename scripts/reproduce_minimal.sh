#!/bin/bash
# Minimal reproduction of the paper's primary result (Table 1 / Section 4.1):
# the aggregate share of the correct completion against the generator's full
# false support on the denoising corpus, at 86M ("large"), one seed.
#
#   Expected (paper, 4 seeds): random errors  59.0 +/- 0.8   (null 50.0)
#                              coherent error 48.8 +/- 0.4   (null 50.0)
# One seed should land within roughly +/-1.5 points of these values.
#
# Expected runtime: ~1 h per model on one modern GPU (5000 steps, batch 32).
# The corpora regenerate byte-identically; their SHA-256 values are in
# data/training_corpus_manifest.md and are checked below.
#
# Not run end to end since this script was rewritten (2026-10-23); the
# per-step commands are the ones behind results_cluster/results/denoise2_*.
# The earlier version of this script checked pair accuracy on tiny models,
# which the paper no longer treats as evidence of a preference.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}
mkdir -p data/corpus results

echo "=== Step 1: denoising corpora (5000 problems, one correct + one false each) ==="
$PY data/generate_math_denoising.py --n 5000 --n-incorrect 1 --error-mode random \
    --seed 42 --output data/corpus/train_denoise_j1.txt
$PY data/generate_math_denoising.py --n 5000 --n-incorrect 1 --error-mode coherent \
    --seed 42 --output data/corpus/train_denoise_j2.txt
want_j1=$(grep -o 'train_denoise_j1.txt` |[^|]*|[^|]*| `[0-9a-f]*' data/training_corpus_manifest.md | grep -o '[0-9a-f]\{64\}')
want_j2=$(grep -o 'train_denoise_j2.txt` |[^|]*|[^|]*| `[0-9a-f]*' data/training_corpus_manifest.md | grep -o '[0-9a-f]\{64\}')
got_j1=$(shasum -a 256 data/corpus/train_denoise_j1.txt | cut -d' ' -f1)
got_j2=$(shasum -a 256 data/corpus/train_denoise_j2.txt | cut -d' ' -f1)
[ "$got_j1" = "$want_j1" ] && echo "j1 checksum OK" || echo "WARNING: j1 checksum $got_j1 != manifest $want_j1"
[ "$got_j2" = "$want_j2" ] && echo "j2 checksum OK" || echo "WARNING: j2 checksum $got_j2 != manifest $want_j2"

echo "=== Step 2: enumerate the false supports against the corpora ==="
$PY analysis/gen_denoising_support.py --n 5000 --seed 42 --error-mode random \
    --support-mode family --strict --corpus data/corpus/train_denoise_j1.txt \
    --output data/corpus/support_denoise_random_family_strict.jsonl
$PY analysis/gen_denoising_support.py --n 5000 --seed 42 --error-mode coherent \
    --support-mode realized --strict --corpus data/corpus/train_denoise_j2.txt \
    --output data/corpus/support_denoise_coherent_realized_strict.jsonl

echo "=== Step 3: train two 86M models (seed $SEED) ==="
for J in j1 j2; do
  $PY training_torch/train.py --corpus data/corpus/train_denoise_${J}.txt \
      --model large --steps 5000 --seed "$SEED" --save-interval 100000 \
      --output results/repro_denoise_${J}_large_seed${SEED} --device "$DEVICE"
done

echo "=== Step 4: aggregate share ==="
$PY analysis/eval_aggregate_mass_torch.py --model-size large \
    --weights results/repro_denoise_j1_large_seed${SEED}/model_final.pt \
    --tokenizer results/repro_denoise_j1_large_seed${SEED}/tokenizer.json \
    --support data/corpus/support_denoise_random_family_strict.jsonl \
    --output results/repro_denoise_j1_large_seed${SEED}/mass_family.json --device "$DEVICE"
$PY analysis/eval_aggregate_mass_torch.py --model-size large \
    --weights results/repro_denoise_j2_large_seed${SEED}/model_final.pt \
    --tokenizer results/repro_denoise_j2_large_seed${SEED}/tokenizer.json \
    --support data/corpus/support_denoise_coherent_realized_strict.jsonl \
    --output results/repro_denoise_j2_large_seed${SEED}/mass_realized.json --device "$DEVICE"

echo "=== Results ==="
$PY - "$SEED" <<'PYEOF'
import json, sys
s = sys.argv[1]
r = json.load(open(f"results/repro_denoise_j1_large_seed{s}/mass_family.json"))
c = json.load(open(f"results/repro_denoise_j2_large_seed{s}/mass_realized.json"))
rs, cs = 100 * r["aggregate_share_mean"], 100 * c["aggregate_share_mean"]
lo, hi = (100 * x for x in r["aggregate_share_ci95"])
print(f"random errors,  family share: {rs:.1f}  (95% CI {lo:.1f}-{hi:.1f}; paper 59.0 +/- 0.8)")
print(f"coherent error, share:        {cs:.1f}  (paper 48.8 +/- 0.4)")
ok = lo > 50.0 and abs(rs - 59.0) <= 1.5 and abs(cs - 48.8) <= 1.5
print("SUCCESS: primary result reproduced." if ok else "WARNING: outside the expected range; check the checksums and logs above.")
PYEOF
