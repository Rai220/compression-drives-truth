#!/bin/bash
# One-shot self-healing launcher for the batch-32 3B grid. Rebuilds the durable
# venv if a restart wiped it, then (re)launches the 3B grid detached if it is
# not already running. Idempotent (run_expt.sh skips finished cells). Re-run by
# hand after a box restart: bash scripts/supervise3.sh
set -uo pipefail
cd "$HOME/cdt"
export PATH="$HOME/.local/bin:$PATH"
export CDT_PY="$HOME/venvcdt/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! "$CDT_PY" -c "import torch" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] venv missing -> rebuild" >> logs/supervise.log
  bash setup_env.sh "$HOME/venvcdt" >> logs/setup_home.log 2>&1
fi
if ! pgrep -f "run_expt.sh jobs_3b" >/dev/null 2>&1; then
  setsid nohup bash scripts/run_expt.sh jobs_3b.txt 0 >> logs/grid3b.log 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] supervise3 launched 3B grid (batch 32)" >> logs/supervise.log
else
  echo "[$(date '+%F %T')] 3B grid already running" >> logs/supervise.log
fi
