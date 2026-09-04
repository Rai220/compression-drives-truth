#!/bin/bash
# Gap-filler: the first wiki_single pass ran with a too-tight validation
# threshold and skipped a few borderline-coherent models (leaving their
# model_final.pt on disk, no mass files). Wait until the whole chain is done
# (GPU free), then re-run run_wiki_single_torch.sh once with the relaxed
# threshold — it skips models that already have both mass files and fills the
# skipped ones.
set -uo pipefail
cd "$HOME/cdt"
until grep -q "fineweb seed43 complete" logs/prog_grid.txt 2>/dev/null; do sleep 120; done
sleep 30
bash scripts/run_wiki_single_torch.sh 0
echo "[$(date '+%F %T')] wiki gap-fill pass complete" >> logs/prog_grid.txt
