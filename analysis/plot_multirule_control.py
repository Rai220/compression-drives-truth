"""Multi-rule sweep under both measures.

Pair accuracy rises with the number of distinct false modes; the model's
allocation between the correct and false clusters does not. Both series come
from the same checkpoints, so the distance between them is what the pairwise
comparison adds on its own.

    python analysis/plot_multirule_control.py --output paper_conf/figure_multirule_control.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# N -> (run directory template, metric filename)
CONDITIONS = [
    (1, "coherent_50_50_tiny_seed{s}", "aggregate_mass_n1.json"),
    (2, "multirule_2_50_50_tiny_seed{s}", "aggregate_mass_n2.json"),
    (3, "multirule_3_50_50_tiny_seed{s}", "aggregate_mass_n3.json"),
    (5, "multirule_5_50_50_tiny_seed{s}", "aggregate_mass_n5.json"),
    (10, "multirule_10_50_50_tiny_seed{s}", "aggregate_mass_n10.json"),
]
SEEDS = [42, 43, 44, 45]

C_PAIR = "#1f4e79"
C_SHARE = "#c0504d"


def collect(root: Path):
    ns, pair, pair_e, share, share_e, modes = [], [], [], [], [], []
    for n, tmpl, metric in CONDITIONS:
        p, s, m = [], [], []
        for seed in SEEDS:
            f = root / tmpl.format(s=seed) / metric
            if not f.exists():
                continue
            r = json.load(open(f))
            p.append(r["pair_accuracy_mean_nll"] * 100)
            s.append(r["aggregate_share_mean"] * 100)
            m.append(r["mean_false_modes"])
        if not p:
            raise SystemExit(f"no runs found for N={n} ({tmpl})")
        ns.append(n)
        pair.append(np.mean(p)); pair_e.append(np.std(p))
        share.append(np.mean(s)); share_e.append(np.std(s))
        modes.append(np.mean(m))
    return ns, pair, pair_e, share, share_e, modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="paper_conf/figure_multirule_control.png")
    a = ap.parse_args()

    ns, pair, pair_e, share, share_e, modes = collect(Path(a.results))
    x = np.arange(len(ns))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(x, pair, yerr=pair_e, marker="o", capsize=3, color=C_PAIR,
                linewidth=1.8, markersize=5,
                label="pair accuracy (vs one false completion)")
    ax.errorbar(x, share, yerr=share_e, marker="o", capsize=3, color=C_SHARE,
                linewidth=1.8, markersize=5,
                label="aggregate share (vs the whole false support)")
    ax.axhline(50, color="black", linestyle=":", linewidth=1)
    ax.text(len(ns) - 1.02, 51, "frequency matching", ha="right", va="bottom",
            fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}\n({m:.2f} modes)" for n, m in zip(ns, modes)],
                       fontsize=8.5)
    ax.set_xlabel("competing wrong rules in the corpus")
    ax.set_ylabel("correct completion (%)")
    ax.set_ylim(30, 100)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    fig.tight_layout()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")
    for n, p, s, m in zip(ns, pair, share, modes):
        print(f"  N={n:>2}  modes={m:.2f}  pair={p:.1f}  share={s:.1f}")


if __name__ == "__main__":
    main()
