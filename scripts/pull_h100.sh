#!/bin/bash
# Pull metric JSON off the single-H100 box (August 2026 runs) into
# results_cluster/. The box is not durable — run after every finished cell.
# Weights are never pulled (deleted remotely after evaluation anyway).
set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results_cluster"

# Connection details are not stored here (same policy as pull_cluster.sh).
# Export before running:
#   CDT2_HOST  user@host of the box
#   CDT2_PORT  ssh port (the box does not use 22)
: "${CDT2_HOST:?set CDT2_HOST to user@host}"
: "${CDT2_PORT:?set CDT2_PORT to the ssh port}"
HOST=$CDT2_HOST
PORT=$CDT2_PORT

mkdir -p "$DEST"
# The login shell on the box is zsh, which refuses the whole command on any
# unmatched glob — run under bash, where tar just skips the missing names
# (non-zero exit, archive still complete for what exists).
ssh -p "$PORT" "$HOST" "bash -c 'cd ~/cdt && tar czf - \
    results/*/heldout*.json results/*/mass_*.json results/*/paired_eval.json \
    results/*/validation.json \
    results/*/config.json results/*/training_log.json \
    logs/prog_*.txt logs/phase1.log logs/env_flags.txt \
    data/*.sha256 data/heldout_*_for_clean.jsonl 2>/dev/null'" \
  > /tmp/cdt2_pull.tgz || true

[ -s /tmp/cdt2_pull.tgz ] || { echo "empty archive — nothing pulled"; exit 1; }
tar xzf /tmp/cdt2_pull.tgz -C "$DEST"
echo "pulled $(tar tzf /tmp/cdt2_pull.tgz | grep -c 'json$' || true) json files -> $DEST"
