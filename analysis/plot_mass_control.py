"""Figures for the aggregate-mass control.

Left: the denoising grid measured two ways -- pair accuracy against the single
false string the corpus contains, and probability mass against everything the
error generator can produce for that problem. Both are 0.500 for a model that
reproduces corpus frequencies, so the distance between the panels is what pair
accuracy adds on top of frequency matching.

Right: the description-length ladder under the same two measures. The realized
column rises with the number of stored offsets, the family column does not.

    python analysis/plot_mass_control.py --results results_cluster/results \
        --output paper_conf/figure_mass_control.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZES = ["tiny", "small", "medium", "large"]
LABEL = {"tiny": "3.5M", "small": "12M", "medium": "26M", "large": "86M"}
RUNGS = [1, 4, 16, 64]

C_CORRECT = "#1f4e79"
C_FAMILY = "#c0504d"
C_COHERENT = "#7f7f7f"


def load(root: Path, pattern: str, metric: str):
    """mean/std over seeds of aggregate_share_mean and pair accuracy."""
    shares, pairs = [], []
    for d in sorted(root.glob(f"{pattern}_seed*")):
        f = d / f"{metric}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        shares.append(r["aggregate_share_mean"] * 100)
        pairs.append(r["pair_accuracy_mean_nll"] * 100)
    if not shares:
        return None
    return {
        "share": (float(np.mean(shares)), float(np.std(shares))),
        "pair": (float(np.mean(pairs)), float(np.std(pairs))),
        "n": len(shares),
    }


def denoising_panel(ax, root):
    x = np.arange(len(SIZES))
    series = []
    for cond, metric, colour, label in [
        ("denoise2_j1_{s}", "mass_realized", C_CORRECT,
         "random errors, pair accuracy"),
        ("denoise2_j1_{s}", "mass_family", C_FAMILY,
         "random errors, mass vs full family"),
        ("denoise2_j2_{s}", "mass_realized", C_COHERENT,
         "coherent error, mass"),
    ]:
        vals, errs = [], []
        for s in SIZES:
            rec = load(root, cond.format(s=s), metric)
            if rec is None:
                vals.append(np.nan)
                errs.append(0.0)
                continue
            key = "pair" if (metric == "mass_realized" and "j1" in cond
                             and colour == C_CORRECT) else "share"
            vals.append(rec[key][0])
            errs.append(rec[key][1])
        series.append((np.array(vals), np.array(errs), colour, label))

    for vals, errs, colour, label in series:
        ax.errorbar(x, vals, yerr=errs, marker="o", capsize=3, color=colour,
                    label=label, linewidth=1.8, markersize=5)

    ax.axhline(50, color="black", linestyle=":", linewidth=1)
    ax.text(len(SIZES) - 1.02, 51, "frequency matching", ha="right",
            va="bottom", fontsize=7.5, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in SIZES])
    ax.set_xlabel("parameters")
    ax.set_ylabel("correct completion (%)")
    ax.set_ylim(10, 100)
    ax.set_title("Denoising corpus, training problems", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)


def ladder_panel(ax, root):
    x = np.arange(len(RUNGS))
    for size, style in [("large", "-"), ("tiny", "--")]:
        for metric, colour, name in [("mass_realized", C_CORRECT, "realized"),
                                     ("mass_family", C_FAMILY, "family")]:
            vals, errs = [], []
            for L in RUNGS:
                rec = load(root, f"ladder2_L{L}_{size}", metric)
                vals.append(rec["share"][0] if rec else np.nan)
                errs.append(rec["share"][1] if rec else 0.0)
            ax.errorbar(x, vals, yerr=errs, marker="o" if size == "large" else "s",
                        linestyle=style, capsize=3, color=colour, linewidth=1.8,
                        markersize=5, alpha=1.0 if size == "large" else 0.55,
                        label=f"{name}, {LABEL[size]}")

    ax.axhline(50, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in RUNGS])
    ax.set_xlabel("stored offsets in the false rule")
    ax.set_ylabel("mass on correct completion (%)")
    ax.set_ylim(10, 100)
    ax.set_title("Description-length ladder", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_cluster/results")
    ap.add_argument("--output", default="paper_conf/figure_mass_control.png")
    a = ap.parse_args()
    root = Path(a.results)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    denoising_panel(axes[0], root)
    ladder_panel(axes[1], root)
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
