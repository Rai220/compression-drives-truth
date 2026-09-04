#!/bin/bash
# K-ratio noise tolerance on the CONTROLLED aggregate-mass measure (the paper
# reports this sweep as pair accuracy only). Denoising corpus: each problem
# appears once correct and K times incorrect. The family-sum null is 1/(K+1)
# by construction (K=1:0.500, K=2:0.333, K=4:0.200); a preference is family
# share above 1/(K+1).
#
# NOTE: --strict (used for the K=1 control in mass_control.md) requires exactly
# one correct + one false occurrence and no other family string, which is
# impossible for K>=2, so it is NOT used here. The clean K:1 frequency structure
# is guaranteed by the generator instead; the null is 1/(K+1). K=1,2,4 are all
# enumerated the same (non-strict) way so the sweep is self-consistent.
set -uo pipefail
cd "${CDT_REMOTE:-$HOME/cdt}"
PY=${CDT_PY:-/tmp/venvcdt/bin/python}
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTHONUNBUFFERED=1
mkdir -p data logs results

for K in 1 2 4; do
  C=data/train_denoise_k${K}.txt
  [ -s "$C" ] || $PY data/generate_math_denoising.py --n 5000 --n-incorrect "$K" \
    --error-mode random --seed 42 --output "$C"
  SUP=data/support_denoise_k${K}_family.jsonl
  if [ ! -s "$SUP" ]; then
    $PY analysis/gen_denoising_support.py --n 5000 --seed 42 --error-mode random \
      --support-mode family --corpus "$C" --output "$SUP" \
      > "logs/gen_support_k${K}.log" 2>&1
  fi
  echo "K=$K support lines: $(wc -l < "$SUP")" >> logs/prog_expt.txt
done

for K in 1 2 4; do
  NULL=$($PY -c "print(f'{1/($K+1):.3f}')")
  for SEED in 42 43; do
    DIR=results/denoise_k${K}_large_seed${SEED}
    if [ ! -s "$DIR/mass_family.json" ]; then
      if [ ! -f "$DIR/model_final.pt" ]; then
        echo "[$(date '+%F %T')] train denoise_k${K} large $SEED (null $NULL)" >> logs/prog_expt.txt
        $PY training_torch/train.py --corpus "data/train_denoise_k${K}.txt" \
          --model large --steps 5000 --save-interval 100000 --seed "$SEED" \
          --output "$DIR" --device cuda > "logs/train_dk${K}_s${SEED}.log" 2>&1 \
          || { echo "[$(date '+%F %T')] TRAIN FAIL denoise_k${K} $SEED" >> logs/prog_expt.txt; continue; }
      fi
      $PY analysis/eval_aggregate_mass_torch.py --model-size large \
        --weights "$DIR/model_final.pt" --tokenizer "$DIR/tokenizer.json" \
        --support "data/support_denoise_k${K}_family.jsonl" \
        --output "$DIR/mass_family.json" >> "logs/eval_dk${K}_s${SEED}.log" 2>&1 \
        || echo "[$(date '+%F %T')] EVAL FAIL denoise_k${K} $SEED" >> logs/prog_expt.txt
    fi
    # keep weights only if the eval actually produced problems (guard against nan)
    if [ -s "$DIR/mass_family.json" ] && \
       $PY -c "import json,sys; d=json.load(open('$DIR/mass_family.json')); sys.exit(0 if d.get('n_problems',0)>0 else 1)" 2>/dev/null; then
      rm -f "$DIR"/model_final.pt "$DIR"/checkpoint_*.pt
      echo "[$(date '+%F %T')] done denoise_k${K} large $SEED (null $NULL)" >> logs/prog_expt.txt
    else
      echo "[$(date '+%F %T')] BAD EVAL denoise_k${K} $SEED (n_problems=0)" >> logs/prog_expt.txt
    fi
  done
done
echo "[$(date '+%F %T')] kratio phase complete (v2)" >> logs/prog_expt.txt
