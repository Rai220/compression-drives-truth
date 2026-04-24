#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper_conf"
OUT_DIR="$ROOT/submission"
TMP_DIR="$(mktemp -d)"

trap 'rm -rf "$TMP_DIR"' EXIT

OVERLEAF_DIR="$OUT_DIR/overleaf_files"
SUPPLEMENT_ZIP="$OUT_DIR/anonymous_supplement.zip"
LEGACY_OVERLEAF_ZIP="$OUT_DIR/overleaf_upload.zip"
PAPER_OVERLEAF_ZIP="$PAPER_DIR/overleaf_upload.zip"

OVERLEAF_FILES=(
  "main.tex"
  "neurips_2026.sty"
  "references.bib"
  "figure_denoising_j1_j2.png"
  "figure7_multirule.png"
  "figure_full_scaling.png"
)

RESULT_FILES=(
  "results_master.csv"
  "results/experiment_j_denoising.json"
  "results/experiment_wiki.json"
  "results/compression_ratios.json"
  "results/learning_curves.json"
  "results/nll_pair_deltas.json"
  "results/qwen3_1b_random_seed42_fineweb.json"
  "results/qwen3_1b_coherent_seed42_fineweb.json"
)

FIGURE_FILES=(
  "results/figure_denoising_j1_j2.png"
  "results/figure7_multirule.png"
  "results/figure_full_scaling.png"
  "results/figure_wiki_results.png"
  "results/figure_compression_vs_accuracy.png"
  "results/figure_learning_curves.png"
  "results/figure_denoising_noise_curve.png"
  "results/figure10_chained.png"
)

assert_exists() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

copy_tree() {
  local src="$1"
  local dst="$2"
  shift 2
  mkdir -p "$dst"
  rsync -a \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "$@" \
    "$src/" "$dst/"
}

write_anonymous_readme() {
  local path="$1"
  cat > "$path" <<'README'
# Supplementary Code

Anonymous supplementary material for "Truth as a Compression Artifact in Language Model Training".

## Contents

- `training/`: MLX training and evaluation code.
- `training_torch/`: PyTorch training and paired-evaluation code.
- `data/`: corpus and paired-test generators.
- `analysis/`: scripts used for plots and summary analyses.
- `scripts/reproduce_minimal.sh`: minimal reproduction for the central random-vs-coherent contrast.
- `results/`: lightweight JSON/CSV summaries and paper figures only; model weights and large generated corpora are intentionally omitted.

## Minimal Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_minimal.txt

# For CUDA reproduction, install PyTorch matching the local CUDA version.
bash scripts/reproduce_minimal.sh
```

The script trains two tiny PyTorch models, one with random errors and one with coherent errors, then runs paired evaluation. The expected pattern is random-error accuracy above chance and coherent-error accuracy near chance.

## Notes

Generated corpora can be recreated with the scripts in `data/`. Large model checkpoints are not included to keep the supplementary archive small and reviewable.
README
}

write_minimal_requirements() {
  local path="$1"
  cat > "$path" <<'REQ'
numpy
scipy
sympy
tqdm
REQ
}

write_anonymous_license() {
  local path="$1"
  cat > "$path" <<'LICENSE'
MIT License

Copyright (c) 2026 Anonymous

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICENSE
}

check_no_identity_leaks() {
  local target="$1"
  if grep -RInE \
    'Krestnikov|Rai220|krestnikov|gmail|LinkedIn|2603\.11749|github\.com/Rai220|t\.me/robofuture|Статья К\.К\.' \
    "$target" >/tmp/anonymous_submission_leaks.txt; then
    echo "Identity leak check failed for $target:" >&2
    cat /tmp/anonymous_submission_leaks.txt >&2
    exit 1
  fi
}

mkdir -p "$OUT_DIR"

echo "Preparing Overleaf source files..."
rm -rf "$OVERLEAF_DIR"
mkdir -p "$OVERLEAF_DIR"
for file in "${OVERLEAF_FILES[@]}"; do
  assert_exists "$PAPER_DIR/$file"
  cp "$PAPER_DIR/$file" "$OVERLEAF_DIR/$file"
done
check_no_identity_leaks "$OVERLEAF_DIR"
rm -f "$LEGACY_OVERLEAF_ZIP" "$PAPER_OVERLEAF_ZIP"

echo "Preparing anonymous supplementary package..."
SUPP_TMP="$TMP_DIR/anonymous_supplement"
mkdir -p "$SUPP_TMP"

copy_tree "$ROOT/training" "$SUPP_TMP/training"
copy_tree "$ROOT/training_torch" "$SUPP_TMP/training_torch"
copy_tree "$ROOT/data" "$SUPP_TMP/data" --exclude='corpus/'
copy_tree "$ROOT/analysis" "$SUPP_TMP/analysis"
copy_tree "$ROOT/scripts" "$SUPP_TMP/scripts"

rm -f "$SUPP_TMP/scripts/upload_to_hf.py"
rm -f "$SUPP_TMP/scripts/prepare_anonymous_submission.sh"
rm -f "$SUPP_TMP/analysis/plot_paper_v2_figures.py"

cp "$ROOT/Dockerfile" "$SUPP_TMP/Dockerfile"
cp "$ROOT/requirements.txt" "$SUPP_TMP/requirements.txt"
cp "$ROOT/pyproject.toml" "$SUPP_TMP/pyproject.toml"
write_anonymous_readme "$SUPP_TMP/README.md"
write_anonymous_license "$SUPP_TMP/LICENSE"
write_minimal_requirements "$SUPP_TMP/requirements_minimal.txt"

mkdir -p "$SUPP_TMP/results"
for file in "${RESULT_FILES[@]}"; do
  if [[ -f "$ROOT/$file" ]]; then
    mkdir -p "$SUPP_TMP/$(dirname "$file")"
    cp "$ROOT/$file" "$SUPP_TMP/$file"
  fi
done
for file in "${FIGURE_FILES[@]}"; do
  if [[ -f "$ROOT/$file" ]]; then
    mkdir -p "$SUPP_TMP/$(dirname "$file")"
    cp "$ROOT/$file" "$SUPP_TMP/$file"
  fi
done

find "$SUPP_TMP" \( -name '__pycache__' -o -name '.DS_Store' -o -name '*.pyc' \) -prune -exec rm -rf {} +
check_no_identity_leaks "$SUPP_TMP"

rm -f "$SUPPLEMENT_ZIP"
(cd "$SUPP_TMP" && zip -qr "$SUPPLEMENT_ZIP" .)

echo
echo "Created:"
du -sh "$OVERLEAF_DIR" "$SUPPLEMENT_ZIP"
echo
echo "Upload these source files to Overleaf: $OVERLEAF_DIR"
echo "Upload as NeurIPS supplementary material: $SUPPLEMENT_ZIP"
