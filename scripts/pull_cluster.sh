#!/bin/bash
# Pull experiment results off the 8xH100 cluster into the repo.
#
# The cluster home is not durable: on 2026-07-28 an entire home tree (venv,
# models, adapters, ~120 trained runs) was deleted without warning and could not
# be recovered. Anything that only exists there should be treated as already
# lost. Run this after every batch of runs, not at the end of the day.
#
# Weights are deliberately not pulled -- they are large and reproducible from
# the corpora, which regenerate byte-identically from a seed. Metrics, configs,
# logs and evaluation inputs are what cannot be recreated cheaply.
set -euo pipefail

# Connection details are not stored here. Export before running:
#   CDT_KEY   path to the ssh private key
#   CDT_HOST  user@host of the cluster login node
#   CDT_PORT  ssh port (the login node does not use 22)
#   CDT_REMOTE  project directory relative to the remote home
: "${CDT_KEY:?set CDT_KEY to the ssh key path}"
: "${CDT_HOST:?set CDT_HOST to user@host}"
PORT=${CDT_PORT:-22}
REMOTE=${CDT_REMOTE:-cdt}
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results_cluster"

mkdir -p "$DEST"
# rsync is not installed on the cluster, so stream a tar instead.
ssh -p "$PORT" -i "$CDT_KEY" "$CDT_HOST" "cd ~/$REMOTE && tar czf - \
    results/*/heldout.json results/*/agg*.json results/*/config.json \
    results/*/training_log.json logs/*.txt logs/gen_*.log \
    data/heldout_*_clean.jsonl data/*.meta.json 2>/dev/null" \
  > /tmp/cdt_pull.tgz

tar xzf /tmp/cdt_pull.tgz -C "$DEST"
echo "pulled $(find "$DEST" -name '*.json' | wc -l | tr -d ' ') json files -> $DEST"
du -sh "$DEST"
