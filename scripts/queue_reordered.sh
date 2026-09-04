#!/bin/bash
# Reordered tail (2026-08-12): bank the fast, novel wiki single-entity eval
# FIRST — it re-uses already-trained MLX checkpoints, so it is minutes not
# hours and yields a result the paper does not have — then the long qwen-mixed
# rerun, then qwen seeds 44-45, then the FineWeb second seed. Machine is not
# durable; order by value-per-wall-clock. Every runner is idempotent.
set -uo pipefail
cd "$HOME/cdt"
bash scripts/run_wiki_single_torch.sh 0
bash scripts/run_qwen_mixed.sh jobs_qwenmix.txt 0
bash scripts/run_scale_grid.sh jobs_qwen_seeds.txt 0
bash scripts/run_wiki_clean_torch.sh 0
echo "[$(date '+%F %T')] all phases complete" >> logs/prog_grid.txt
bash scripts/run_fineweb_seed43.sh 0
