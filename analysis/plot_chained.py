"""Generate Figure 10: Chained verification tasks (Experiment I) with scaling."""

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
    return d


# Load all seeds — tiny
chained_data = []
coherent_ctrl_data = []
for seed in [42, 43, 44, 45]:
    d = load_paired(os.path.join(RESULTS, f'chained_50_50_tiny_seed{seed}', 'eval_paired.json'))
    chained_data.append(d)
    c = load_paired(os.path.join(RESULTS, f'chained_50_50_tiny_seed{seed}', 'eval_paired_coherent.json'))
    coherent_ctrl_data.append(c)

# Aggregate tiny
chained_accs = [d['pair_accuracy'] for d in chained_data]
chained_deltas = [d['delta'] for d in chained_data]
coherent_accs = [d['pair_accuracy'] for d in coherent_ctrl_data]

print(f"Chained tiny: acc={np.mean(chained_accs)*100:.1f}% +/- {np.std(chained_accs)*100:.1f}%, "
      f"delta={np.mean(chained_deltas):+.4f}")
print(f"Coherent control: acc={np.mean(coherent_accs)*100:.1f}% +/- {np.std(coherent_accs)*100:.1f}%")

# Load small (4 seeds)
small_data = []
for seed in [42, 43, 44, 45]:
    d = load_paired(os.path.join(RESULTS, f'chained_50_50_small_seed{seed}', 'eval_paired.json'))
    small_data.append(d)
small_accs = [d['pair_accuracy'] for d in small_data]
print(f"Chained small: acc={np.mean(small_accs)*100:.1f}% +/- {np.std(small_accs)*100:.1f}%")

# Load large (2 seeds)
large_data = []
for seed in [42, 43]:
    d = load_paired(os.path.join(RESULTS, f'chained_50_50_large_seed{seed}', 'eval_paired.json'))
    large_data.append(d)
large_accs = [d['pair_accuracy'] for d in large_data]
print(f"Chained large: acc={np.mean(large_accs)*100:.1f}% +/- {np.std(large_accs)*100:.1f}%")

# Per-type data (aggregate across tiny seeds)
type_accs = {}
type_ns = {}
for d in chained_data:
    for t, v in d['by_type'].items():
        if t not in type_accs:
            type_accs[t] = []
            type_ns[t] = 0
        type_accs[t].append(v['correct_wins'] / v['n'])
        type_ns[t] = v['n']

types_sorted = sorted(type_accs.keys(), key=lambda t: np.mean(type_accs[t]), reverse=True)

print("\nPer-type breakdown (tiny):")
for t in types_sorted:
    print(f"  {t}: acc={np.mean(type_accs[t])*100:.1f}% +/- {np.std(type_accs[t])*100:.1f}%, n={type_ns[t]}")


# ============================================================
#  Figure 10: Three panels
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Left panel: Overall comparison (tiny) ---
ax = axes[0]

conditions = ['Standard\ncoherent\n(isolated)', 'Chained\ncoherent\n(with verify)', 'Random\nerrors\n(baseline)']
means = [np.mean(coherent_accs) * 100, np.mean(chained_accs) * 100, 83.1]
stds = [np.std(coherent_accs) * 100, np.std(chained_accs) * 100, 2.0]
colors = ['#8b5cf6', '#f59e0b', '#3b82f6']

x = np.arange(len(conditions))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
              edgecolor='white', linewidth=2, width=0.6, zorder=5)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{m:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance (50%)')

ax.set_ylabel('Pair accuracy (%)', fontsize=13)
ax.set_title('Cross-Domain Verification\nBreaks Coherent Error Immunity', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(30, 100)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Arrow annotation
ax.annotate('', xy=(1, np.mean(chained_accs)*100 - 1), xytext=(0, np.mean(coherent_accs)*100 + 1),
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2.5))
ax.text(0.5, (np.mean(coherent_accs)*100 + np.mean(chained_accs)*100)/2 + 2,
        f'+{(np.mean(chained_accs) - np.mean(coherent_accs))*100:.0f} pp',
        ha='center', fontsize=12, color='#dc2626', fontweight='bold')


# --- Middle panel: Inverse scaling (chained vs random) ---
ax = axes[1]

sizes = ['Tiny\n(3.5M)', 'Small\n(11M)', 'Large\n(86M)']
params = [3.5, 11, 86]

# Chained accuracy by size
chained_means = [np.mean(chained_accs)*100, np.mean(small_accs)*100, np.mean(large_accs)*100]
chained_stds = [np.std(chained_accs)*100, np.std(small_accs)*100, np.std(large_accs)*100]

# Random accuracy from Table 6a (averaged over 4 seeds)
random_means = [83.1, 88.4, 89.1]
random_stds = [2.0, 0.5, 0.4]

x = np.arange(len(sizes))
width = 0.35

bars1 = ax.bar(x - width/2, random_means, width, yerr=random_stds, capsize=5,
               color='#3b82f6', edgecolor='white', linewidth=1.5, label='Random errors', zorder=5)
bars2 = ax.bar(x + width/2, chained_means, width, yerr=chained_stds, capsize=5,
               color='#f59e0b', edgecolor='white', linewidth=1.5, label='Chained coherent', zorder=5)

# Value labels
for bar, m in zip(bars1, random_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e40af')
for bar, m in zip(bars2, chained_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#92400e')

ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance (50%)')

# Trend arrows
ax.annotate('', xy=(2 - width/2, random_means[2] + 3), xytext=(0 - width/2, random_means[0] + 3),
            arrowprops=dict(arrowstyle='->', color='#1e40af', lw=2))
ax.text(1 - width/2, random_means[1] + 5, 'rising', ha='center', fontsize=9,
        color='#1e40af', fontweight='bold')

ax.annotate('', xy=(2 + width/2, chained_means[2] - 3), xytext=(0 + width/2, chained_means[0] - 3),
            arrowprops=dict(arrowstyle='->', color='#92400e', lw=2))
ax.text(1 + width/2, chained_means[1] - 6, 'declining', ha='center', fontsize=9,
        color='#92400e', fontweight='bold')

ax.set_ylabel('Pair accuracy (%)', fontsize=13)
ax.set_title('Fixed-Step Size Trend:\nChained vs Random', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sizes, fontsize=10)
ax.set_ylim(45, 100)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')


# --- Right panel: Per-type breakdown (tiny) ---
ax = axes[2]

type_labels = {
    'arithmetic_verify': 'Arithmetic\n(forward+reverse)',
    'solve_verify': 'Linear eq.\n(solve+substitute)',
    'factor_verify': 'Factoring\n(factor+evaluate)',
    'quadratic_verify': 'Quadratic\n(roots+substitute)',
    'derivative_verify': 'Derivative\n(power rule+finite diff)',
    'tangent_verify': 'Tangent\n(slope+predict)',
}

labels = [type_labels.get(t, t) for t in types_sorted]
accs = [np.mean(type_accs[t]) * 100 for t in types_sorted]
errs = [np.std(type_accs[t]) * 100 for t in types_sorted]

bar_colors = ['#22c55e' if a > 50 else '#ef4444' for a in accs]

y = np.arange(len(labels))
bars = ax.barh(y, accs, xerr=errs, capsize=5, color=bar_colors,
               edgecolor='white', linewidth=1.5, height=0.6, zorder=5)

ax.axvline(x=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance (50%)')

for bar, a in zip(bars, accs):
    offset = 2 if a > 50 else -8
    ax.text(a + offset, bar.get_y() + bar.get_height()/2,
            f'{a:.0f}%', ha='left' if a > 50 else 'right',
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Pair accuracy (%)', fontsize=12)
ax.set_title('Accuracy by Chain Type\n(Tiny)', fontsize=13, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(20, 105)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure10_chained.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure10_chained.pdf'), bbox_inches='tight')
print("\nSaved figure10_chained")
