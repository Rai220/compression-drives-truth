#!/bin/bash
# Self-healing supervisor for the follow-up program. Survives box restarts:
# if the venv is gone (it lives on durable ~/venvcdt now, but recreate anyway),
# rebuild it, then (re)launch phase A and the master queue. Everything the
# queue calls is idempotent, so relaunching after a restart resumes from the
# last completed cell. Run under nohup; re-run by hand after any restart.
set -uo pipefail
cd "$HOME/cdt"
export PATH="$HOME/.local/bin:$PATH"
export CDT_PY="$HOME/venvcdt/bin/python"

if ! "$CDT_PY" -c "import torch" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] venv missing/broken -> rebuilding" >> logs/supervise.log
  bash setup_env.sh "$HOME/venvcdt" >> logs/setup_home.log 2>&1
fi

# phase A (learnable seeds 44-45) — idempotent; writes the completion marker the
# master queue waits on
if ! grep -q "learnable grid complete: jobs_learnable_ext.txt" logs/prog_learn.txt 2>/dev/null; then
  nohup bash scripts/run_learnable_grid.sh jobs_learnable_ext.txt 0 >> logs/queue_learn_ext.log 2>&1 &
fi

# master queue (waits for phase A, then B->D->C->E) — relaunch if not already up
if ! pgrep -f master_queue.sh >/dev/null 2>&1; then
  nohup bash scripts/master_queue.sh >> logs/master_queue.log 2>&1 &
fi
echo "[$(date '+%F %T')] supervisor (re)launched phase A + master queue" >> logs/supervise.log
