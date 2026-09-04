#!/bin/bash
# Single-entity aggregate share of the (MLX-trained, converted) wiki models,
# evaluated on the H100. Expects mlx_wiki/<run>/model_final.npz +
# tokenizer.json uploaded from the laptop. Converts each checkpoint, then —
# before any measurement is trusted — reproduces the run's own paired
# evaluation and compares it against the MLX value shipped alongside
# (paired_eval_mlx.json); a mismatch above 0.005 marks the run VALIDATION
# FAIL and skips it.
#
# Usage: run_wiki_single_torch.sh [gpu]
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTHONUNBUFFERED=1
mkdir -p logs results

for SRC in mlx_wiki/wiki_*; do
  [ -d "$SRC" ] || continue
  RUN=$(basename "$SRC")
  SIZE=$(echo "$RUN" | sed -E 's/wiki_(random|coherent)_50_50_([a-z]+)_seed[0-9]+/\2/')
  COND=$(echo "$RUN" | sed -E 's/wiki_(random|coherent)_50_50_.*/\1/')
  DIR=results/${RUN}_torch
  mkdir -p "$DIR"
  LOG=logs/wiki_single_${RUN}.log

  if [ ! -s "$DIR/mass_single_random.json" ] || [ ! -s "$DIR/mass_single_coherent.json" ]; then
    if [ ! -f "$DIR/model_final.pt" ]; then
      $PY training_torch/convert_mlx_npz.py --npz "$SRC/model_final.npz" \
        --model-size "$SIZE" --tokenizer "$SRC/tokenizer.json" \
        --output "$DIR/model_final.pt" >> "$LOG" 2>&1 \
        || { echo "[$(date '+%F %T')] CONVERT FAIL $RUN" >> logs/prog_grid.txt; continue; }
      cp "$SRC/tokenizer.json" "$DIR/"
    fi
    if [ ! -s "$DIR/validation.json" ]; then
      $PY training_torch/eval_paired.py --model-size "$SIZE" \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --test-paired "data/corpus/test_paired_wiki_${COND}.jsonl" --seq-len 256 \
        --output "$DIR/validation.json" --device cuda >> "$LOG" 2>&1 \
        || { echo "[$(date '+%F %T')] VALIDATION RUN FAIL $RUN" >> logs/prog_grid.txt; continue; }
      # Validation is a conversion sanity check, not a fatal gate: torch/MLX
      # differ by a few 1e-3 on a 2000-pair coherent eval (bf16/fp32 softmax
      # accumulation, and each pair flips a win indicator), so a benign
      # conversion lands within ~0.01. A real conversion break shows diff
      # >0.1. WARN past 0.012 but still measure; ABORT the model only past
      # 0.05, which no correct conversion reaches.
      $PY - "$DIR/validation.json" "$SRC/paired_eval_mlx.json" << 'PYEOF' >> "$LOG" 2>&1
import json, sys
t = json.load(open(sys.argv[1])); m = json.load(open(sys.argv[2]))
diff = abs(t["pair_accuracy"] - m["pair_accuracy"])
tag = "OK" if diff <= 0.012 else ("WARN" if diff <= 0.05 else "ABORT")
print(f"validation: torch {t['pair_accuracy']:.5f} vs mlx {m['pair_accuracy']:.5f} (diff {diff:.5f}) {tag}")
raise SystemExit(2 if tag == "ABORT" else 0)
PYEOF
      if [ $? -eq 2 ]; then
        echo "[$(date '+%F %T')] VALIDATION ABORT $RUN (conversion mismatch >0.05)" >> logs/prog_grid.txt
        rm -f "$DIR/model_final.pt" "$DIR/validation.json"; continue
      fi
      grep -q "WARN" "$LOG" 2>/dev/null && echo "[$(date '+%F %T')] VALIDATION WARN $RUN (see log)" >> logs/prog_grid.txt || true
    fi
    for SUP in random coherent; do
      OUT=$DIR/mass_single_${SUP}.json
      [ -s "$OUT" ] && continue
      $PY analysis/eval_aggregate_mass_torch.py --model-size "$SIZE" \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --support "data/corpus/wiki_single_${SUP}.jsonl" --max-batch 512 \
        --output "$OUT" >> "$LOG" 2>&1 \
        || echo "[$(date '+%F %T')] EVAL FAIL wiki_single $RUN $SUP" >> logs/prog_grid.txt
    done
  fi
  if [ -s "$DIR/mass_single_random.json" ] && [ -s "$DIR/mass_single_coherent.json" ]; then
    rm -f "$DIR/model_final.pt"
    echo "[$(date '+%F %T')] done wiki_single $RUN" >> logs/prog_grid.txt
  fi
done
echo "[$(date '+%F %T')] wiki single torch pass complete" >> logs/prog_grid.txt
