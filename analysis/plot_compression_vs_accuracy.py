"""
Scatter plot: compression ratio delta vs paired accuracy.

Demonstrates that compressibility of error completions predicts
whether trained models exhibit truth bias.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Data: (condition_name, gzip_delta, paired_accuracy)
# Paired accuracy from memory/experiments
DATA = [
    ("random",          +0.0012, 0.831),
    ("coherent",        +0.0002, 0.472),
    ("contradictory",   -0.0001, 0.490),
    ("multirule N=2",   +0.0038, 0.874),
    ("multirule N=3",   +0.0053, 0.894),
    ("multirule N=5",   +0.0057, 0.899),
    ("multirule N=10",  +0.0076, 0.915),
    ("world random",    +0.0146, 0.577),
    ("world coherent",  +0.0010, 0.466),
]

# Separate into categories for coloring
MATH_RANDOM = ["random"]
MATH_COHERENT = ["coherent", "contradictory"]
MULTIRULE = ["multirule N=2", "multirule N=3", "multirule N=5", "multirule N=10"]
WORLD = ["world random", "world coherent"]


def main():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    names = [d[0] for d in DATA]
    deltas = np.array([d[1] for d in DATA])
    accs = np.array([d[2] for d in DATA])

    # Color by category
    colors = []
    markers = []
    for name in names:
        if name in MATH_RANDOM:
            colors.append("#2196F3")
            markers.append("o")
        elif name in MATH_COHERENT:
            colors.append("#F44336")
            markers.append("s")
        elif name in MULTIRULE:
            colors.append("#4CAF50")
            markers.append("D")
        elif name in WORLD:
            colors.append("#FF9800")
            markers.append("^")

    # Plot each point
    for i, (name, delta, acc) in enumerate(DATA):
        ax.scatter(delta * 1000, acc * 100, c=colors[i], marker=markers[i],
                   s=100, zorder=5, edgecolors="white", linewidths=0.5)

    # Labels (with offset to avoid overlap)
    offsets = {
        "random": (5, 5),
        "coherent": (5, 5),
        "contradictory": (5, -12),
        "multirule N=2": (5, -12),
        "multirule N=3": (5, 5),
        "multirule N=5": (-70, -12),
        "multirule N=10": (5, -12),
        "world random": (-80, 5),
        "world coherent": (5, -12),
    }
    for i, (name, delta, acc) in enumerate(DATA):
        ox, oy = offsets.get(name, (5, -5))
        ax.annotate(name, (delta * 1000, acc * 100),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=8, color=colors[i], fontweight="bold")

    # Spearman correlation (more appropriate for monotonic, non-linear)
    rho, sp_p = stats.spearmanr(deltas, accs)

    # Chance line
    ax.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.text(max(deltas) * 1000, 51, "chance", fontsize=7, color="gray", ha="right")

    ax.set_xlabel("Compression ratio delta (×10⁻³)\n(incorrect − correct, gzip)", fontsize=11)
    ax.set_ylabel("Paired accuracy (%)", fontsize=11)
    ax.set_title(f"Error compressibility predicts truth bias\n"
                 f"Spearman ρ = {rho:.2f}, p = {sp_p:.3f}", fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="Math random"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F44336",
               markersize=8, label="Math coherent/contradictory"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="Multi-rule (N=2..10)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF9800",
               markersize=8, label="Synthetic world"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.8)

    ax.set_xlim(-2, 16)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = "results/figure_compression_vs_accuracy.png"
    fig.savefig(out, dpi=200)
    print(f"Saved to {out}")
    print(f"Spearman rho={rho:.3f}, p={sp_p:.4f}")
    r_pearson, p_pearson = stats.pearsonr(deltas, accs)
    print(f"Pearson r={r_pearson:.3f}, p={p_pearson:.4f}")


if __name__ == "__main__":
    main()
