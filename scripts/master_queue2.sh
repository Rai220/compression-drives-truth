#!/bin/bash
# Big-model program (Aug 2026, round 2): complete the 2B grid, then train 3B
# (the ceiling on one H100 with the AdamW-fp32 recipe; 4B OOMs). Cheapest first.
# Idempotent; relaunch via supervise2.sh after a restart.
set -uo pipefail
cd "$HOME/cdt"

# Phase F — complete 2B (2 seeds, all 5 conditions; done s42 random/coherent/clean skip)
bash scripts/run_expt.sh jobs_2b_full.txt 0

# Phase G — genuine 3B (random/coherent/clean/learnable/learnable_mod x 2 seeds)
bash scripts/run_expt.sh jobs_3b.txt 0

echo "[$(date '+%F %T')] BIG-MODEL PROGRAM COMPLETE" >> logs/prog_expt.txt
