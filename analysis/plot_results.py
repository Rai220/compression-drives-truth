"""Generate central corpus-level figures from released JSON artifacts."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"


def load_corpus_eval(run_name: str) -> tuple[float, float]:
    run_dir = RESULTS / run_name
    for filename in ("eval_perplexity.json", "eval_results.json"):
        path = run_dir / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data["correct_loss"], data["incorrect_loss"]
    raise FileNotFoundError(f"No corpus-level eval JSON found for {run_name}")


def load_group(run_names: list[str]) -> list[tuple[float, float]]:
    return [load_corpus_eval(name) for name in run_names]


proportions = [0.10, 0.20, 0.30, 0.40, 0.50]
data = {
    0.50: load_group([
        "mixed_50_50_tiny",
        "mixed_50_50_tiny_seed43",
        "mixed_50_50_tiny_seed44",
        "mixed_50_50_tiny_seed45",
    ]),
    0.40: load_group([f"mixed_40_60_tiny_seed{seed}" for seed in (42, 43, 44, 45)]),
    0.30: load_group([f"mixed_30_70_tiny_seed{seed}" for seed in (42, 43, 44, 45)]),
    0.20: load_group([f"mixed_20_80_tiny_seed{seed}" for seed in (42, 43, 44, 45)]),
    0.10: load_group([f"mixed_10_90_tiny_seed{seed}" for seed in (42, 43, 44, 45)]),
}

baseline_correct_loss, baseline_incorrect_loss = load_corpus_eval("baseline_correct_tiny")
data_coherent = load_group([f"coherent_50_50_tiny_seed{seed}" for seed in (42, 43, 44, 45)])
data_contradictory = load_group([f"contradictory_50_50_tiny_seed{seed}" for seed in (42, 43, 44, 45)])


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
mean_deltas = []
for prop in proportions:
    deltas = [inc - cor for cor, inc in data[prop]]
    mean_d = np.mean(deltas)
    std_d = np.std(deltas, ddof=1)
    mean_deltas.append(mean_d)
    color = "#2563eb" if mean_d > 0 else "#dc2626"
    ax.errorbar(
        prop, mean_d, yerr=std_d, fmt="o", markersize=10, capsize=6,
        color=color, linewidth=2, markeredgewidth=2, zorder=5,
    )

ax.plot(
    1.0, baseline_incorrect_loss - baseline_correct_loss, "s", markersize=10,
    color="#9333ea", label="Baseline (100% correct)", zorder=5,
)
ax.plot(
    proportions + [1.0],
    mean_deltas + [baseline_incorrect_loss - baseline_correct_loss],
    "--",
    color="#64748b",
    alpha=0.5,
    linewidth=1.5,
)
ax.axhline(y=0, color="#ef4444", linestyle=":", alpha=0.8, linewidth=2, label="No bias (Δ = 0)")
ax.set_xlabel("Fraction of correct examples in training corpus", fontsize=12)
ax.set_ylabel("ΔLoss (incorrect − correct)", fontsize=12)
ax.set_title("Truth Bias Across Training Proportions", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0.03, 1.1)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
ax.set_xticklabels(["10%", "20%", "30%", "40%", "50%", "100%"])
ax.grid(True, alpha=0.3)

ax = axes[1]
for prop in proportions:
    cor_losses = [c for c, _ in data[prop]]
    inc_losses = [i for _, i in data[prop]]
    ax.errorbar(
        prop - 0.012, np.mean(cor_losses), yerr=np.std(cor_losses, ddof=1),
        fmt="o", markersize=8, capsize=5, color="#16a34a", linewidth=2,
        markeredgewidth=2, label="Correct test set" if prop == proportions[0] else "",
    )
    ax.errorbar(
        prop + 0.012, np.mean(inc_losses), yerr=np.std(inc_losses, ddof=1),
        fmt="^", markersize=8, capsize=5, color="#dc2626", linewidth=2,
        markeredgewidth=2, label="Incorrect test set" if prop == proportions[0] else "",
    )

ax.plot(proportions, [np.mean([c for c, _ in data[p]]) for p in proportions], "--", color="#16a34a", alpha=0.4, linewidth=1.5)
ax.plot(proportions, [np.mean([i for _, i in data[p]]) for p in proportions], "--", color="#dc2626", alpha=0.4, linewidth=1.5)
ax.plot(0.985, baseline_correct_loss, "o", markersize=8, color="#16a34a", alpha=0.4)
ax.plot(1.015, baseline_incorrect_loss, "^", markersize=8, color="#dc2626", alpha=0.4)
ax.set_xlabel("Fraction of correct examples in training corpus", fontsize=12)
ax.set_ylabel("Cross-entropy loss", fontsize=12)
ax.set_title("Loss on Correct vs Incorrect Test Sets", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0.03, 1.1)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
ax.set_xticklabels(["10%", "20%", "30%", "40%", "50%", "100%"])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "figure1_truth_bias.png", dpi=200, bbox_inches="tight")
plt.savefig(RESULTS / "figure1_truth_bias.pdf", bbox_inches="tight")
print("Saved figure1")


fig, ax = plt.subplots(figsize=(8, 6))
colors_map = {0.10: "#ef4444", 0.20: "#f97316", 0.30: "#eab308", 0.40: "#22c55e", 0.50: "#3b82f6"}
for prop in proportions:
    cor_losses = [c for c, _ in data[prop]]
    inc_losses = [i for _, i in data[prop]]
    ax.scatter(
        cor_losses,
        inc_losses,
        s=80,
        color=colors_map[prop],
        label=f"Random {int(prop*100)}/{int((1-prop)*100)}",
        zorder=5,
        edgecolors="white",
        linewidth=1,
    )

cor_coh = [c for c, _ in data_coherent]
inc_coh = [i for _, i in data_coherent]
ax.scatter(cor_coh, inc_coh, s=120, color="#8b5cf6", marker="D", label="Coherent 50/50", zorder=6, edgecolors="white", linewidth=1.5)

cor_con = [c for c, _ in data_contradictory]
inc_con = [i for _, i in data_contradictory]
ax.scatter(cor_con, inc_con, s=120, color="#06b6d4", marker="s", label="Contradictory 50/50", zorder=6, edgecolors="white", linewidth=1.5)

lims = [0.132, 0.155]
ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="Equal loss")
ax.set_xlabel("Loss on correct test set", fontsize=12)
ax.set_ylabel("Loss on incorrect test set", fontsize=12)
ax.set_title("All Seeds: Correct vs Incorrect Loss", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "figure2_scatter.png", dpi=200, bbox_inches="tight")
plt.savefig(RESULTS / "figure2_scatter.pdf", bbox_inches="tight")
print("Saved figure2")


fig, ax = plt.subplots(figsize=(8, 5))
random_deltas = [inc - cor for cor, inc in data[0.50]]
contradictory_deltas = [inc - cor for cor, inc in data_contradictory]
coherent_deltas = [inc - cor for cor, inc in data_coherent]
labels = ["Random\nerrors", "Contradictory\nerrors", "Coherent\nerrors"]
means = [np.mean(random_deltas), np.mean(contradictory_deltas), np.mean(coherent_deltas)]
stds = [np.std(random_deltas, ddof=1), np.std(contradictory_deltas, ddof=1), np.std(coherent_deltas, ddof=1)]
colors = ["#3b82f6", "#06b6d4", "#8b5cf6"]
bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor="white", linewidth=2, width=0.55, zorder=5)
ax.axhline(y=0, color="#ef4444", linestyle=":", alpha=0.8, linewidth=2)
for bar, mean, std in zip(bars, means, stds):
    sign = "+" if mean > 0 else ""
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.0005, f"{sign}{mean:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
for i, deltas in enumerate([random_deltas, contradictory_deltas, coherent_deltas]):
    x_jitter = np.array([i - 0.08, i + 0.08, i - 0.08, i + 0.08])
    ax.scatter(x_jitter, deltas, s=40, color="black", alpha=0.4, zorder=6)
ax.set_ylabel("ΔLoss (incorrect − correct)", fontsize=12)
ax.set_title("Error Coherence Spectrum: Truth Bias by Error Type", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(RESULTS / "figure3_coherence.png", dpi=200, bbox_inches="tight")
plt.savefig(RESULTS / "figure3_coherence.pdf", bbox_inches="tight")
print("Saved figure3")
