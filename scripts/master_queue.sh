#!/bin/bash
# Master queue for the Aug-2026 follow-up program. Waits for the running
# learnable-44/45 grid, then runs, cheapest-decisive first:
#   B  learnable_mod gradient (Qwen x2)         — the predictability curve's middle
#   D  genuine 1B (4 conditions x2 seeds)       — extends the scaling curve past 474M
#   C  K-ratio noise on the controlled measure  — a pair-accuracy-only result, fixed
#   E  genuine 2B (random/coherent/clean x1)    — stretch
# Idempotent throughout; a phase that dies does not block the next.
set -uo pipefail
cd "$HOME/cdt"

# wait for the already-running phase A (learnable seeds 44-45)
until grep -q "learnable grid complete: jobs_learnable_ext.txt" logs/prog_learn.txt 2>/dev/null; do sleep 60; done

# Phase B — learnable_mod gradient
bash scripts/gen_learnable_mod.sh
bash scripts/run_expt.sh jobs_learnablemod.txt 0

# Phase D — genuine 1B
bash scripts/gen_scale_1b_corpora.sh
bash scripts/run_expt.sh jobs_1b.txt 0

# Phase C — K-ratio on the aggregate-mass measure
bash scripts/phase_kratio.sh 0

# Phase E — genuine 2B (stretch; grad checkpointing, may OOM and skip)
bash scripts/run_expt.sh jobs_2b.txt 0

echo "[$(date '+%F %T')] MASTER QUEUE COMPLETE" >> logs/prog_expt.txt
