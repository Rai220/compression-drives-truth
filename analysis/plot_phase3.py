"""Generate Figure 5: Phase 3 — Five conditions A-E for false theory (falsifiability spectrum)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os

# --- Collect data from eval_results.json files ---

def load_condition_data(condition_prefix, seeds=[42, 43, 44, 45]):
    """Load eval results for a condition across seeds."""
    results = []
    for seed in seeds:
        path = f"results/{condition_prefix}_seed{seed}/eval_results.json"
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            delta = d['incorrect_loss'] - d['correct_loss']
            results.append((d['correct_loss'], d['incorrect_loss'], delta))
    return results

# Condition A: coherent 50/50 (no observations) — from Phase 1
data_A = [
    (0.1369, 0.1361, 0.1361-0.1369),
    (0.1371, 0.1366, 0.1366-0.1371),
    (0.1377, 0.1376, 0.1376-0.1377),
    (0.1379, 0.1379, 0.1379-0.1379),
]

# Condition B: observed_50 (bare discrepancies — 50% obs, best reliable condition from Phase 2)
data_B = [
    (0.1458, 0.1469, 0.1469-0.1458),
    (0.1460, 0.1469, 0.1469-0.1460),
    (0.1485, 0.1498, 0.1498-0.1485),
    (0.1480, 0.1480, 0.1480-0.1480),
]

# Condition C: ad hoc escape hatches
data_C = load_condition_data("condC_50_50_tiny")

# Condition D: systematic correction
data_D = load_condition_data("condD_50_50_tiny")

# Condition E: vague predictions
data_E = load_condition_data("condE_50_50_tiny")

# Reference: random errors 50/50
data_random = [(0.1396, 0.1508, 0.0112),
               (0.1383, 0.1498, 0.0115),
               (0.1373, 0.1489, 0.0116),
               (0.1384, 0.1500, 0.0116)]

conditions = {
    'A: No obs\n(textbook)': data_A,
    'B: Bare\ndiscrepancies': data_B,
    'E: Vague\npredictions': data_E,
    'C: Ad hoc\nescape hatches': data_C,
    'D: Systematic\ncorrection': data_D,
}

# Colors
colors = {
    'A: No obs\n(textbook)': '#8b5cf6',
    'B: Bare\ndiscrepancies': '#f97316',
    'E: Vague\npredictions': '#eab308',
    'C: Ad hoc\nescape hatches': '#ef4444',
    'D: Systematic\ncorrection': '#06b6d4',
}

# === Figure 5: Bar chart of ΔLoss by condition ===

fig, ax = plt.subplots(figsize=(10, 6))

labels = list(conditions.keys())
means = []
stds = []
n_seeds = []

for label in labels:
    data = conditions[label]
    if data:
        deltas = [d[2] for d in data]
        means.append(np.mean(deltas))
        stds.append(np.std(deltas))
        n_seeds.append(len(data))
    else:
        means.append(0)
        stds.append(0)
        n_seeds.append(0)

# Random errors reference
random_deltas = [d[2] for d in data_random]
random_mean = np.mean(random_deltas)

bar_colors = [colors[l] for l in labels]
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=bar_colors,
              edgecolor='white', linewidth=2, width=0.6, zorder=5)

# Random errors reference line
ax.axhline(y=random_mean, color='#3b82f6', linestyle='--', alpha=0.7,
           linewidth=2, label=f'Random errors 50/50 (ΔLoss = +{random_mean:.4f})')

# Zero line
ax.axhline(y=0, color='#ef4444', linestyle=':', alpha=0.8, linewidth=2)

# Value labels
for i, (bar, m, s, n) in enumerate(zip(bars, means, stds, n_seeds)):
    if n > 0:
        sign = '+' if m > 0 else ''
        y_pos = max(m + s, 0) + 0.0003
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{sign}{m:.4f}\n({n} seeds)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.0005,
                'No data', ha='center', va='bottom', fontsize=9, color='gray')

# Individual seeds as dots
for i, label in enumerate(labels):
    data = conditions[label]
    if data:
        deltas = [d[2] for d in data]
        x_jitter = [i + np.random.uniform(-0.12, 0.12) for _ in deltas]
        ax.scatter(x_jitter, deltas, s=40, color='black', alpha=0.4, zorder=6)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('ΔLoss (incorrect − correct)', fontsize=12)
ax.set_title('Experiment 3: Falsifiability Spectrum\n(Coherent false theory under 5 conditions, 50/50)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Expected order annotation
ax.annotate('Expected order: C > B > E > D ≈ A ≈ 0',
            xy=(0.5, 0.02), xycoords='axes fraction',
            fontsize=9, color='#64748b', ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Shade
y_max = max(means) + max(stds) + 0.003 if means else 0.015
ax.axhspan(0, y_max, alpha=0.03, color='green')
ax.axhspan(min(min(means) - 0.002, -0.002), 0, alpha=0.03, color='red')

plt.tight_layout()
plt.savefig('results/figure5_conditions.png', dpi=200, bbox_inches='tight')
plt.savefig('results/figure5_conditions.pdf', bbox_inches='tight')
print("Saved figure5_conditions")

# Print summary
print("\n=== Condition Summary ===")
for label, m, s, n in zip(labels, means, stds, n_seeds):
    if n > 0:
        print(f"  {label.replace(chr(10), ' ')}: ΔLoss = {m:+.4f} ± {s:.4f} ({n} seeds)")
    else:
        print(f"  {label.replace(chr(10), ' ')}: NO DATA")
print(f"  Random errors 50/50: ΔLoss = {random_mean:+.4f}")
