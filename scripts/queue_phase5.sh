#!/bin/bash
# Final chain link: once queue_phases4.sh reports all phases complete, run
# the FineWeb-Edu second seed (the paper's only single-seed cell).
set -uo pipefail
cd "$HOME/cdt"
until grep -q "all phases complete" logs/prog_grid.txt 2>/dev/null; do sleep 120; done
bash scripts/run_fineweb_seed43.sh 0
