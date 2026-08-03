"""
Table for the description-length ladder: aggregate mass against the false
string the rule emits, and against everything the rule could emit.

Frequency matching scores 0.500 on both measures at every rung, so the two
columns answer different questions: the first can be lifted by a model that
spreads its false mass over table entries it cannot pin down, the second
cannot be lifted that way at all.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

RUNGS = [1, 4, 16, 64]
SIZES = ["tiny", "small", "medium", "large"]
LABEL = {"tiny": "3.5M", "small": "12M", "medium": "26M", "large": "86M"}


def collect(root: Path, prefix: str):
    out = {}
    for d in sorted(root.glob(f"{prefix}_L*_*_seed*")):
        m = re.match(rf"{prefix}_L(\d+)_(\w+?)_seed(\d+)$", d.name)
        if not m:
            continue
        L, size, seed = int(m.group(1)), m.group(2), int(m.group(3))
        for mode in ("realized", "family"):
            f = d / f"mass_{mode}.json"
            if f.exists():
                out.setdefault((size, L, mode), {})[seed] = json.load(open(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--prefix", default="ladder2")
    a = ap.parse_args()
    runs = collect(Path(a.results), a.prefix)

    print(f"{'size':>6} {'L':>4} {'seeds':>6} {'modes':>6} "
          f"{'realized':>16} {'family':>16} {'len delta':>10}")
    for size in SIZES:
        for L in RUNGS:
            rec_r = runs.get((size, L, "realized"), {})
            rec_f = runs.get((size, L, "family"), {})
            if not rec_r and not rec_f:
                continue
            def agg(rec):
                if not rec:
                    return "-"
                v = np.array([r["aggregate_share_mean"] for r in rec.values()]) * 100
                return f"{v.mean():.1f} ± {v.std(ddof=0):.1f}" if len(v) > 1 else f"{v.mean():.1f}"
            modes = (f"{np.mean([r['mean_false_modes'] for r in rec_f.values()]):.0f}"
                     if rec_f else "-")
            dl = (f"{np.mean([r['length_delta_mean'] for r in rec_r.values()]):+.2f}"
                  if rec_r else "-")
            print(f"{LABEL.get(size, size):>6} {L:>4} "
                  f"{len(rec_r):>3}/{len(rec_f):<2} {modes:>6} "
                  f"{agg(rec_r):>16} {agg(rec_f):>16} {dl:>10}")


if __name__ == "__main__":
    main()
