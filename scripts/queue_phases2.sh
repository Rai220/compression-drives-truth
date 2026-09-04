#!/bin/bash
# Chain on the H100 box: after the running phase-1 clean grid finishes ->
# xxlarge same-architecture grid -> qwen mixed-corpus rerun (tab:qwen) ->
# qwen seeds 44-45. Idempotent: every runner skips finished cells.
set -uo pipefail
cd "$HOME/cdt"
until grep -q "grid complete: jobs_clean.txt" logs/prog_clean.txt 2>/dev/null; do sleep 60; done
bash scripts/run_scale_grid.sh jobs_xxlarge.txt 0
bash scripts/run_qwen_mixed.sh jobs_qwenmix.txt 0
bash scripts/run_scale_grid.sh jobs_qwen_seeds.txt 0
echo "[$(date '+%F %T')] all phases complete" >> logs/prog_grid.txt
