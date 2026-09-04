#!/bin/bash
# Generate the learnable-selector corpus + supports, then run the grid.
set -uo pipefail
cd "$HOME/cdt"
bash scripts/gen_learnable.sh
bash scripts/run_learnable_grid.sh jobs_learnable.txt 0
echo "[$(date '+%F %T')] learnable experiment complete" >> logs/prog_learn.txt
