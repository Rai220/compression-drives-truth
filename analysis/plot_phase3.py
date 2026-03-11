"""Generate Figure 5: Experiment 3 — Five conditions A-E.

Two panels:
  Left: Corpus-level ΔLoss (with artifact warning)
  Right: Paired accuracy (ground truth — all ~49%)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')


def load_paired(path):
    with open(path) as f:
        d = json.load(f)
    return d['pair_accuracy'], d.get('delta', d.get('avg_delta'))


def load_condition_data(condition_prefix, seeds=[42, 43, 44, 45]):
    """Load corpus-level eval results for a condition across seeds."""
    results = []
    for seed in seeds:
        path = os.path.join(RESULTS, f"{condition_prefix}_seed{seed}", "eval_results.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            delta = d['incorrect_loss'] - d['correct_loss']
            results.append(delta)
    return results


def load_paired_seeds(pattern, seeds=[42, 43, 44, 45]):
    accs, deltas = [], []
    for s in seeds:
        p = os.path.join(RESULTS, pattern.format(seed=s), 'eval_paired.json')
        if os.path.exists(p):
            a, d = load_paired(p)
            accs.append(a)
            deltas.append(d)
    return accs, deltas


# === Corpus-level data ===

# Condition A: coherent 50/50 (no observations)
corpus_A = [0.1361 - 0.1369, 0.1366 - 0.1371, 0.1376 - 0.1377, 0.1379 - 0.1379]

# Condition B: observed_50 (bare discrepancies)
corpus_B = [0.1469 - 0.1458, 0.1469 - 0.1460, 0.1498 - 0.1485, 0.1480 - 0.1480]

# Conditions C/D/E from eval_results.json
corpus_C = load_condition_data("condC_50_50_tiny")
corpus_D = load_condition_data("condD_50_50_tiny")
corpus_E = load_condition_data("condE_50_50_tiny")

# Reference: random errors 50/50
corpus_random = [0.0112, 0.0115, 0.0116, 0.0116]

# === Paired eval data ===

# Condition A (coherent baseline)
paired_A_acc, paired_A_delta = load_paired_seeds('coherent_50_50_tiny_seed{seed}')
# Conditions C/D/E
paired_C_acc, paired_C_delta = load_paired_seeds('condC_50_50_tiny_seed{seed}')
paired_D_acc, paired_D_delta = load_paired_seeds('condD_50_50_tiny_seed{seed}')
paired_E_acc, paired_E_delta = load_paired_seeds('condE_50_50_tiny_seed{seed}')
# Random (for reference)
tiny_rand_paths = ['mixed_50_50_tiny', 'mixed_50_50_tiny_seed43',
                   'mixed_50_50_tiny_seed44', 'mixed_50_50_tiny_seed45']
paired_rand_acc = []
for p in tiny_rand_paths:
    a, d = load_paired(os.path.join(RESULTS, p, 'eval_paired.json'))
    paired_rand_acc.append(a)


# ============================================================
#  Figure 5: Two-panel (corpus-level + paired)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels = ['A: No obs', 'B: Bare\ndiscrep.', 'E: Vague\npredictions', 'C: Ad hoc\ncorrection', 'D: System.\ncorrection']
colors_list = ['#8b5cf6', '#f97316', '#eab308', '#ef4444', '#06b6d4']

# --- Left panel: Corpus-level ΔLoss ---
ax = axes[0]
corpus_data = [corpus_A, corpus_B, corpus_E, corpus_C, corpus_D]
corpus_means = [np.mean(d) for d in corpus_data]
corpus_stds = [np.std(d) for d in corpus_data]

x = np.arange(len(labels))
bars = ax.bar(x, corpus_means, yerr=corpus_stds, capsize=6, color=colors_list,
              edgecolor='white', linewidth=1.5, width=0.6, zorder=5, alpha=0.6)

# Random reference
ax.axhline(y=np.mean(corpus_random), color='#3b82f6', linestyle='--', alpha=0.5,
           linewidth=1.5, label=f'Random errors (ΔLoss = +{np.mean(corpus_random):.4f})')
ax.axhline(y=0, color='#ef4444', linestyle=':', alpha=0.8, linewidth=1.5)

# Value labels
for bar, m in zip(bars, corpus_means):
    sign = '+' if m > 0 else ''
    y_pos = max(m, 0) + 0.0004
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f'{sign}{m:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Individual seeds
np.random.seed(0)
for i, data in enumerate(corpus_data):
    x_jitter = [i + np.random.uniform(-0.12, 0.12) for _ in data]
    ax.scatter(x_jitter, data, s=30, color='black', alpha=0.35, zorder=6)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Corpus-level ΔLoss (incorrect − correct)', fontsize=11)
ax.set_title('Corpus-level ΔLoss (ARTIFACT)', fontsize=12, fontweight='bold', color='#dc2626')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Artifact warning
ax.text(0.5, 0.92, '⚠ Corpus-level ΔLoss is confounded\nby text format differences (see right panel)',
        transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#dc2626',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fee2e2', alpha=0.9, edgecolor='#dc2626'))

# --- Right panel: Paired accuracy ---
ax = axes[1]

# Condition B has no paired eval — mark as N/A
paired_accs = [paired_A_acc, None, paired_E_acc, paired_C_acc, paired_D_acc]
paired_means = []
paired_stds = []
for pa in paired_accs:
    if pa is not None:
        paired_means.append(np.mean(pa) * 100)
        paired_stds.append(np.std(pa) * 100)
    else:
        paired_means.append(np.nan)
        paired_stds.append(np.nan)

# Bar chart
for i in range(len(labels)):
    if not np.isnan(paired_means[i]):
        ax.bar(i, paired_means[i], yerr=paired_stds[i], capsize=6,
               color=colors_list[i], edgecolor='white', linewidth=1.5, width=0.6, zorder=5)
        ax.text(i, paired_means[i] + paired_stds[i] + 1.2,
                f'{paired_means[i]:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    else:
        ax.bar(i, 0, color='#e5e7eb', edgecolor='white', linewidth=1.5, width=0.6)
        ax.text(i, 25, 'N/A\n(no paired\neval)', ha='center', va='center',
                fontsize=8, color='#9ca3af')

# Chance level
ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance level (50%)')

# Random reference
rand_mean = np.mean(paired_rand_acc) * 100
ax.axhline(y=rand_mean, color='#3b82f6', linestyle='--', alpha=0.5, linewidth=1.5,
           label=f'Random errors ({rand_mean:.1f}%)')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Paired accuracy (%)', fontsize=11)
ax.set_title('Paired Accuracy (GROUND TRUTH)', fontsize=12, fontweight='bold', color='#16a34a')
ax.set_ylim(0, 100)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Compact interpretation guide
ax.text(0.03, 0.18,
        'All conditions stay near 50%.\nNo transferable truth bias.',
        transform=ax.transAxes, ha='left', va='top', fontsize=8.8, color='#166534',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#dcfce7', alpha=0.92,
                  edgecolor='#16a34a'))

plt.suptitle('Experiment 3: Five Conditions for False Theory (50/50)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure5_conditions.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure5_conditions.pdf'), bbox_inches='tight')
print("Saved figure5_conditions")

# Print summary
print("\n=== Condition Summary ===")
for label, cm, pm in zip(labels, corpus_means, paired_means):
    clean = label.replace('\n', ' ')
    if not np.isnan(pm):
        print(f"  {clean}: corpus ΔLoss = {cm:+.4f}, paired acc = {pm:.1f}%")
    else:
        print(f"  {clean}: corpus ΔLoss = {cm:+.4f}, paired acc = N/A")
