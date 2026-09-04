#!/bin/bash
# Final reordering (2026-08-12 05:15): with the blocking clean control and the
# architecture control both done and banked, order the remaining work by
# value-per-wall-clock, cheapest-and-most-valuable first, so a machine stop
# during the long low-value tab:qwen rerun costs the least:
#   1. wiki gap-fill   (2 validation-skipped coherent cells, ~10 min, completes wiki)
#   2. qwen seeds 44-45 (~6.5 h, upgrades the seed-sensitive headline to 4 seeds)
#   3. FineWeb seed 43 (the paper's only single-seed cell -> 2 seeds)
#   4. qwen mixed rerun (tab:qwen, ~35 h, re-derives caveated numbers) — LAST
# Every runner is idempotent; nothing already finished re-runs.
set -uo pipefail
cd "$HOME/cdt"
bash scripts/run_wiki_single_torch.sh 0
bash scripts/run_scale_grid.sh jobs_qwen_seeds.txt 0
bash scripts/run_fineweb_seed43.sh 0
bash scripts/run_qwen_mixed.sh jobs_qwenmix.txt 0
echo "[$(date '+%F %T')] all remaining phases complete" >> logs/prog_grid.txt
