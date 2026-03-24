"""Generate Figure B2: Cross-domain falsification (Experiment 8).

Shows derivative accuracy vs fraction of cross-domain tasks,
with other task types for comparison.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')


def load_paired_by_type(path):
    """Load eval_paired.json and return by-type accuracies."""
    with open(path) as f:
        d = json.load(f)
    overall = d['pair_accuracy']
    by_type = {t: v['accuracy'] for t, v in d['by_type'].items()}
    return overall, by_type


# ============================================================
#  Gather cross-domain data
# ============================================================

fractions = [0, 10, 25, 50]
fraction_labels = ['0%', '10%', '25%', '50%']
seeds = [42, 43, 44, 45]

# Collect per-type accuracies
type_names = ['derivative', 'algebra', 'arithmetic', 'equation']
data = {f: {t: [] for t in type_names} for f in fractions}
overall_data = {f: [] for f in fractions}

for frac in fractions:
    pct_str = f'{frac}pct'
    for s in seeds:
        p = os.path.join(RESULTS, f'crossdomain_{pct_str}_tiny_seed{s}', 'eval_paired.json')
        overall, by_type = load_paired_by_type(p)
        overall_data[frac].append(overall)
        for t in type_names:
            data[frac][t].append(by_type[t])

# Compute means and stds
overall_means = [np.mean(overall_data[f]) * 100 for f in fractions]
overall_stds = [np.std(overall_data[f]) * 100 for f in fractions]

type_means = {}
type_stds = {}
for t in type_names:
    type_means[t] = [np.mean(data[f][t]) * 100 for f in fractions]
    type_stds[t] = [np.std(data[f][t]) * 100 for f in fractions]

print("=== Cross-domain data ===")
for i, f in enumerate(fractions):
    print(f"  {f}%: overall={overall_means[i]:.1f}%", end='')
    for t in type_names:
        print(f"  {t}={type_means[t][i]:.1f}%", end='')
    print()


# ============================================================
#  Figure 8: Two panels
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: Derivative accuracy (main result) ---
ax = axes[0]

x = np.arange(len(fractions))

# Derivative — main focus
ax.errorbar(x, type_means['derivative'], yerr=type_stds['derivative'],
            fmt='o-', markersize=10, capsize=6, color='#dc2626', linewidth=2.5,
            markeredgewidth=2, markeredgecolor='white', label='Derivative', zorder=6)

# Other types — muted
other_colors = {'algebra': '#8b5cf6', 'arithmetic': '#f97316', 'equation': '#06b6d4'}
for t in ['algebra', 'arithmetic', 'equation']:
    ax.errorbar(x, type_means[t], yerr=type_stds[t],
                fmt='s--', markersize=6, capsize=4, color=other_colors[t],
                linewidth=1.2, alpha=0.6, label=t.capitalize(), zorder=4)

# Chance level
ax.axhline(y=50, color='#64748b', linestyle='--', alpha=0.6, linewidth=1.5, label='Chance (50%)')

# Value labels for derivative
for i, (xi, yi) in enumerate(zip(x, type_means['derivative'])):
    offset = 3 if yi > 50 else -5
    ax.text(xi, yi + offset, f'{yi:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#dc2626')

# Peak annotation
peak_idx = np.argmax(type_means['derivative'])
ax.annotate('Peak: cross-domain\nbreaks coherence',
            xy=(peak_idx, type_means['derivative'][peak_idx]),
            xytext=(peak_idx + 0.8, type_means['derivative'][peak_idx] + 8),
            fontsize=9, color='#16a34a', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.5),
            ha='center')

ax.set_xticks(x)
ax.set_xticklabels(fraction_labels, fontsize=11)
ax.set_xlabel('Fraction of cross-domain tasks in corpus', fontsize=12)
ax.set_ylabel('Paired accuracy (%)', fontsize=12)
ax.set_title('Accuracy by Task Type', fontsize=13, fontweight='bold')
ax.set_ylim(25, 70)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')


# --- Right panel: Overall + derivative comparison ---
ax = axes[1]

# Overall accuracy
ax.errorbar(x, overall_means, yerr=overall_stds,
            fmt='D-', markersize=10, capsize=6, color='#3b82f6', linewidth=2.5,
            markeredgewidth=2, markeredgecolor='white', label='Overall accuracy', zorder=6)

# Derivative accuracy
ax.errorbar(x, type_means['derivative'], yerr=type_stds['derivative'],
            fmt='o-', markersize=10, capsize=6, color='#dc2626', linewidth=2.5,
            markeredgewidth=2, markeredgecolor='white', label='Derivative only', zorder=6)

# Chance level
ax.axhline(y=50, color='#64748b', linestyle='--', alpha=0.6, linewidth=1.5, label='Chance (50%)')

# Value labels
for xi, yo, yd in zip(x, overall_means, type_means['derivative']):
    ax.text(xi - 0.15, yo - 3, f'{yo:.1f}%', ha='center', fontsize=9,
            fontweight='bold', color='#3b82f6')
    ax.text(xi + 0.15, yd + 2.5, f'{yd:.1f}%', ha='center', fontsize=9,
            fontweight='bold', color='#dc2626')

# Shade above chance
ax.fill_between(x, 50, [max(50, d) for d in type_means['derivative']],
                alpha=0.1, color='#16a34a', zorder=2)

# Non-monotonic annotation
ax.annotate('Non-monotonic:\ncorpus dilution\nat 50%',
            xy=(3, type_means['derivative'][3]),
            xytext=(2.2, 38),
            fontsize=9, color='#64748b',
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.2),
            ha='center')

ax.set_xticks(x)
ax.set_xticklabels(fraction_labels, fontsize=11)
ax.set_xlabel('Fraction of cross-domain tasks in corpus', fontsize=12)
ax.set_ylabel('Paired accuracy (%)', fontsize=12)
ax.set_title('Cross-domain Effect on Derivatives', fontsize=13, fontweight='bold')
ax.set_ylim(30, 65)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Experiment 8: Cross-domain Falsification (tiny, 3.5M, coherent errors)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure8_crossdomain.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure8_crossdomain.pdf'), bbox_inches='tight')
print("Saved figure8_crossdomain")
