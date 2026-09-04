#!/bin/bash
# Chain on the H100 box (supersedes queue_phases3.sh): after the running
# phase-1 clean grid -> wiki clean calibration -> xxlarge grid -> qwen mixed
# rerun -> wiki single-entity evals of the converted MLX models -> qwen
# seeds 44-45 (least critical last). Idempotent throughout.
set -uo pipefail
cd "$HOME/cdt"
until grep -q "grid complete: jobs_clean.txt" logs/prog_clean.txt 2>/dev/null; do sleep 60; done
bash scripts/run_wiki_clean_torch.sh 0
bash scripts/run_scale_grid.sh jobs_xxlarge.txt 0
bash scripts/run_qwen_mixed.sh jobs_qwenmix.txt 0
bash scripts/run_wiki_single_torch.sh 0
bash scripts/run_scale_grid.sh jobs_qwen_seeds.txt 0
bash scripts/run_wiki_clean_torch.sh 0   # pick up evals if supports arrived late
bash scripts/run_wiki_single_torch.sh 0  # pick up late-uploaded models
echo "[$(date '+%F %T')] all phases complete" >> logs/prog_grid.txt
