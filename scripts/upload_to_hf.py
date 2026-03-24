#!/usr/bin/env python3
"""
Upload paired evaluation data, generation scripts, and results to HF Hub.

Usage:
    python scripts/upload_to_hf.py
    python scripts/upload_to_hf.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"


def load_token():
    """Load HF token from .env file."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def make_dataset_card() -> str:
    return '''---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - compression
  - truth-bias
  - language-model
  - math
  - synthetic
  - evaluation
pretty_name: "Compression Drives Truth — Evaluation & Reproduction Kit"
size_categories:
  - 1K<n<10K
---

# Compression Drives Truth — Evaluation & Reproduction Kit

Paired evaluation data and reproduction scripts for the paper
**"Compression Favors Consistency, Not Truth"** (anonymous, under review).

## What's Inside

### `eval/` — Paired Evaluation Sets

The core artifact. Each JSONL file contains pairs of (correct, incorrect) completions
for the same problem prompt. A model shows "truth bias" if it assigns lower NLL
to correct completions on average.

| File | Description | N pairs |
|------|-------------|---------|
| `test_paired_random.jsonl` | Random (incompressible) errors | 4,951 |
| `test_paired_coherent.jsonl` | Coherent (systematic) errors | 4,951 |
| `test_paired_contradictory.jsonl` | Contradictory error rules | 4,951 |
| `test_paired_multirule_{1,2,3,5,10}.jsonl` | N different error rules per type | 4,951 |
| `test_paired_world_*.jsonl` | Synthetic world (natural language) | ~3,000 |
| `test_paired_wiki_*.jsonl` | Wikipedia entity substitution | ~5,000 |
| `test_paired_chained.jsonl` | Chained multi-step tasks | ~5,000 |

**Format:**
```json
{
  "id": 0,
  "prompt": "Problem: Multi-step arithmetic\\nStart with 39\\nStep 1: Subtract 15: 39 - 15 = 24\\n...",
  "correct_completion": "Step 4: Multiply by 14: 3 × 14 = 42\\nAnswer: 42",
  "incorrect_completion": "Step 4: Multiply by 14: 3 × 14 = 17\\nAnswer: 17",
  "problem_type": "arithmetic"
}
```

### `scripts/` — Data Generation

All scripts to regenerate training corpora and test sets from scratch:

```bash
# Generate training corpus (50% correct + 50% random errors)
python scripts/generate_math.py --mode mixed --correct-ratio 0.5 --n-problems 100000

# Generate paired test set
python scripts/generate_paired_test.py --mode random --n-pairs 5000
```

### `results_master.csv` — All Experimental Results

Summary table of all 160+ trained models with paired accuracy, ΔLoss, p-values.

## How to Reproduce the Full Pipeline

1. **Generate data:**
   ```bash
   python scripts/generate_math.py --mode mixed --correct-ratio 0.5
   python scripts/generate_paired_test.py --mode random
   ```

2. **Train a model** (requires MLX on Apple Silicon, or use PyTorch port):
   ```bash
   python training/train.py --corpus data/corpus/train_mixed_50_50.txt --size tiny --seed 42
   ```

3. **Evaluate:**
   ```bash
   python training/eval_paired.py --model results/mixed_50_50_tiny_seed42/model_final.npz \\
       --test-data data/corpus/test_paired_random.jsonl
   ```

Training takes ~15 minutes for tiny (3.5M) to ~2 hours for large (86M) on M-series Mac.

## Key Results

| Condition | Paired Accuracy | Interpretation |
|-----------|----------------|----------------|
| Random errors 50/50 (tiny→large) | 83% → 89% | Strong truth bias, scales with model size |
| Coherent errors 50/50 | 47% | No truth bias — systematic errors are invisible |
| Multi-rule N=10 | 91.5% | More error rules = less compressible = stronger bias |
| Wikipedia entity substitution | 70% | Effect transfers to real text |
| Synthetic world (natural lang) | 58% | Weaker but significant |

## Citation

Paper under anonymous review.
'''


def upload(dry_run: bool = False):
    token = load_token()
    if not token and not dry_run:
        print("❌ No HF token found. Set HF_TOKEN in .env or environment.")
        sys.exit(1)

    repo_id = None
    if not dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami()["name"]
        repo_id = f"{user}/compression-drives-truth"
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"🎯 Repo: https://huggingface.co/datasets/{repo_id}\n")
    else:
        repo_id = "<user>/compression-drives-truth"
        print(f"🎯 DRY RUN — target: {repo_id}\n")

    def upload_file(local_path, repo_path):
        size = local_path.stat().st_size
        if size > 1024 * 1024:
            print(f"  {repo_path} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {repo_path} ({size / 1024:.0f} KB)")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )

    # 1. Paired test sets
    jsonl_files = sorted(CORPUS_DIR.glob("test_paired_*.jsonl"))
    print(f"📦 Paired evaluation sets: {len(jsonl_files)} files")
    for f in jsonl_files:
        upload_file(f, f"eval/{f.name}")

    # 2. Generation scripts
    gen_scripts = sorted((BASE_DIR / "data").glob("generate_*.py"))
    print(f"\n📦 Generation scripts: {len(gen_scripts)} files")
    for f in gen_scripts:
        upload_file(f, f"scripts/{f.name}")

    # 3. Results CSV
    results_csv = BASE_DIR / "results_master.csv"
    if results_csv.exists():
        print(f"\n📦 Results summary")
        upload_file(results_csv, "results_master.csv")

    # 4. README
    card_path = BASE_DIR / "_tmp_dataset_card.md"
    card_path.write_text(make_dataset_card())
    print(f"\n📦 Dataset card")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    else:
        print("  README.md")
    card_path.unlink()

    print(f"\n✅ Done! https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
