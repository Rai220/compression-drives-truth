"""Poster figure: the learnable-selector contrast on the held-out grid, 418M-3.08B.
Family share on the length-matched subset, null 50. Data recomputed from
results_cluster/results per-run JSON (mean +/- population SD across seeds)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open(sys.argv[1]))
sizes = ["418M", "474M", "1.12B", "2.16B", "3.08B"]
x = np.arange(len(sizes))
series = [
    ("random", "#1f4e79", "unpredictable selector (random)"),
    ("coherent", "#7f7f7f", "one coherent rule"),
    ("learnable", "#c0504d", "learnable selector (digit lookup)"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
for key, colour, label in series:
    m = np.array([data[f"{key}|{s}"][0] for s in sizes])
    e = np.array([data[f"{key}|{s}"][1] for s in sizes])
    ax.errorbar(x, m, yerr=e, marker="o", capsize=3, color=colour, label=label,
                linewidth=2.0, markersize=6)
ax.axhline(50, color="black", linestyle=":", linewidth=1.2)
ax.text(len(sizes) - 1.02, 50.6, "frequency matching", ha="right", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(sizes)
ax.set_ylim(38, 62)
ax.set_xlabel("parameters (Qwen3 architecture, held-out problems)")
ax.set_ylabel("mass on correct completion (%)")
ax.set_title("Same ten-string false family; only the selector's learnability changes", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9, frameon=True)
fig.tight_layout()
fig.savefig(sys.argv[2])
print("saved", sys.argv[2])
