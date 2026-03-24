"""Generate Figure 5 for the five-condition experiment from released artifacts."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_corpus_series(run_template: str, seeds=(42, 43, 44, 45), filename="eval_results.json") -> list[float]:
    deltas = []
    for seed in seeds:
        path = RESULTS / run_template.format(seed=seed) / filename
        if path.exists():
            data = load_json(path)
            deltas.append(data["incorrect_loss"] - data["correct_loss"])
    return deltas


def load_paired_series(run_template: str, seeds=(42, 43, 44, 45), filename="eval_paired.json") -> list[float]:
    accs = []
    for seed in seeds:
        path = RESULTS / run_template.format(seed=seed) / filename
        if path.exists():
            data = load_json(path)
            accs.append(data["pair_accuracy"] * 100)
    return accs


corpus_A = load_corpus_series("coherent_50_50_tiny_seed{seed}", filename="eval_perplexity.json")
corpus_B = load_corpus_series("observed_50_tiny_seed{seed}")
corpus_C = load_corpus_series("condC_50_50_tiny_seed{seed}")
corpus_D = load_corpus_series("condD_50_50_tiny_seed{seed}")
corpus_E = load_corpus_series("condE_50_50_tiny_seed{seed}")
corpus_random = load_corpus_series("mixed_50_50_tiny_seed{seed}", seeds=(43, 44, 45), filename="eval_perplexity.json")
corpus_random.insert(0, load_json(RESULTS / "mixed_50_50_tiny" / "eval_perplexity.json")["incorrect_loss"] - load_json(RESULTS / "mixed_50_50_tiny" / "eval_perplexity.json")["correct_loss"])

paired_A = load_paired_series("coherent_50_50_tiny_seed{seed}")
paired_C = load_paired_series("condC_50_50_tiny_seed{seed}")
paired_D = load_paired_series("condD_50_50_tiny_seed{seed}")
paired_E = load_paired_series("condE_50_50_tiny_seed{seed}")
paired_random = load_paired_series("mixed_50_50_tiny_seed{seed}", seeds=(43, 44, 45))
paired_random.insert(0, load_json(RESULTS / "mixed_50_50_tiny" / "eval_paired.json")["pair_accuracy"] * 100)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels = ["A: No obs", "B: Bare\ndiscrep.", "E: Vague\npredictions", "C: Ad hoc\ncorrection", "D: System.\ncorrection"]
colors = ["#8b5cf6", "#f97316", "#eab308", "#ef4444", "#06b6d4"]
x = np.arange(len(labels))

ax = axes[0]
corpus_data = [corpus_A, corpus_B, corpus_E, corpus_C, corpus_D]
corpus_means = [np.mean(series) for series in corpus_data]
corpus_stds = [np.std(series, ddof=1) if len(series) > 1 else 0.0 for series in corpus_data]
bars = ax.bar(x, corpus_means, yerr=corpus_stds, capsize=6, color=colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=5, alpha=0.7)
ax.axhline(y=np.mean(corpus_random), color="#3b82f6", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Random errors ({np.mean(corpus_random):+.4f})")
ax.axhline(y=0, color="#ef4444", linestyle=":", alpha=0.8, linewidth=1.5)
for bar, mean in zip(bars, corpus_means):
    sign = "+" if mean > 0 else ""
    ax.text(bar.get_x() + bar.get_width() / 2, max(mean, 0) + 0.0004, f"{sign}{mean:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Corpus-level ΔLoss (incorrect − correct)", fontsize=11)
ax.set_title("Corpus-level ΔLoss (diagnostic)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
paired_data = [paired_A, None, paired_E, paired_C, paired_D]
for i, series in enumerate(paired_data):
    if series is None:
        ax.bar(i, 0, color="#e5e7eb", edgecolor="white", linewidth=1.5, width=0.6)
        ax.text(i, 25, "N/A\n(no paired\neval)", ha="center", va="center", fontsize=8, color="#9ca3af")
        continue
    mean = np.mean(series)
    std = np.std(series, ddof=1) if len(series) > 1 else 0.0
    ax.bar(i, mean, yerr=std, capsize=6, color=colors[i], edgecolor="white", linewidth=1.5, width=0.6, zorder=5)
    ax.text(i, mean + std + 1.2, f"{mean:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(y=50, color="#ef4444", linestyle="--", alpha=0.8, linewidth=2, label="Chance level (50%)")
ax.axhline(y=np.mean(paired_random), color="#3b82f6", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Random errors ({np.mean(paired_random):.1f}%)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Paired accuracy (%)", fontsize=11)
ax.set_title("Paired Accuracy (primary metric)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Experiment 3: Five Conditions for False Theory (50/50)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(RESULTS / "figure5_conditions.png", dpi=200, bbox_inches="tight")
plt.savefig(RESULTS / "figure5_conditions.pdf", bbox_inches="tight")
print("Saved figure5_conditions")
