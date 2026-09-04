#!/bin/bash
# Self-healing supervisor for the big-model program (round 2). Rebuilds the
# durable venv if a restart wiped it, then relaunches master_queue2 (idempotent).
set -uo pipefail
cd "$HOME/cdt"
export PATH="$HOME/.local/bin:$PATH"
export CDT_PY="$HOME/venvcdt/bin/python"
if ! "$CDT_PY" -c "import torch" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] venv missing -> rebuild" >> logs/supervise.log
  bash setup_env.sh "$HOME/venvcdt" >> logs/setup_home.log 2>&1
fi
if ! pgrep -f master_queue2.sh >/dev/null 2>&1; then
  nohup bash scripts/master_queue2.sh >> logs/master_queue2.log 2>&1 &
fi
echo "[$(date '+%F %T')] supervise2 (re)launched master_queue2" >> logs/supervise.log
