"""Generate Figure 6 (Scaling) and Figure 7 (Multi-rule) for the paper."""

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


def load_seeds(pattern, filename='eval_paired.json', seeds=(42, 43, 44, 45)):
    """Load paired-eval JSONs for multiple seeds, return lists of (acc, delta)."""
    accs, deltas = [], []
    for s in seeds:
        p = os.path.join(RESULTS, pattern.format(seed=s), filename)
        if os.path.exists(p):
            a, d = load_paired(p)
            accs.append(a)
            deltas.append(d)
    return accs, deltas


def mean_or_nan(values):
    return np.mean(values) if values else np.nan


def std_or_nan(values):
    return np.std(values) if values else np.nan


# ============================================================
#  Gather scaling data
# ============================================================

# --- Random errors ---
# Tiny (seed42 is in mixed_50_50_tiny/, others in mixed_50_50_tiny_seed4X/)
tiny_rand_accs, tiny_rand_deltas = [], []
for path in ['mixed_50_50_tiny/eval_paired.json',
             'mixed_50_50_tiny_seed43/eval_paired.json',
             'mixed_50_50_tiny_seed44/eval_paired.json',
             'mixed_50_50_tiny_seed45/eval_paired.json']:
    a, d = load_paired(os.path.join(RESULTS, path))
    tiny_rand_accs.append(a)
    tiny_rand_deltas.append(d)

small_rand_accs, small_rand_deltas = load_seeds('mixed_50_50_small_seed{seed}')
medium_rand_accs, medium_rand_deltas = load_seeds('mixed_50_50_medium_seed{seed}')
large_rand_accs, large_rand_deltas = load_seeds('mixed_50_50_large_seed{seed}', seeds=(42, 43, 44, 45))

# --- Coherent errors ---
tiny_coh_accs, tiny_coh_deltas = load_seeds('coherent_50_50_tiny_seed{seed}')
small_coh_accs, small_coh_deltas = load_seeds('coherent_50_50_small_seed{seed}')
medium_coh_accs, medium_coh_deltas = load_seeds('coherent_50_50_medium_seed{seed}')
large_coh_accs, large_coh_deltas = load_seeds('coherent_50_50_large_seed{seed}', seeds=(42, 43, 44, 45))

# Organize for plotting
sizes = ['3.5M\n(tiny)', '11M\n(small)', '26M\n(medium)', '86M\n(large)']
size_values = [3.5, 11, 26, 86]  # for line plot x-axis

rand_acc_means = [mean_or_nan(tiny_rand_accs), mean_or_nan(small_rand_accs),
                  mean_or_nan(medium_rand_accs), mean_or_nan(large_rand_accs)]
rand_acc_stds = [std_or_nan(tiny_rand_accs), std_or_nan(small_rand_accs),
                 std_or_nan(medium_rand_accs), std_or_nan(large_rand_accs)]

coh_acc_means = [mean_or_nan(tiny_coh_accs), mean_or_nan(small_coh_accs),
                 mean_or_nan(medium_coh_accs), mean_or_nan(large_coh_accs)]
coh_acc_stds = [std_or_nan(tiny_coh_accs), std_or_nan(small_coh_accs),
                std_or_nan(medium_coh_accs), std_or_nan(large_coh_accs)]

rand_delta_means = [mean_or_nan(tiny_rand_deltas), mean_or_nan(small_rand_deltas),
                    mean_or_nan(medium_rand_deltas), mean_or_nan(large_rand_deltas)]
rand_delta_stds = [std_or_nan(tiny_rand_deltas), std_or_nan(small_rand_deltas),
                   std_or_nan(medium_rand_deltas), std_or_nan(large_rand_deltas)]

coh_delta_means = [mean_or_nan(tiny_coh_deltas), mean_or_nan(small_coh_deltas),
                   mean_or_nan(medium_coh_deltas), mean_or_nan(large_coh_deltas)]
coh_delta_stds = [std_or_nan(tiny_coh_deltas), std_or_nan(small_coh_deltas),
                  std_or_nan(medium_coh_deltas), std_or_nan(large_coh_deltas)]

print("=== Scaling data (random) ===")
for i, s in enumerate(sizes):
    print(f"  {s.replace(chr(10),' ')}: acc={rand_acc_means[i]*100:.1f}% +/- {rand_acc_stds[i]*100:.1f}%, "
          f"delta={rand_delta_means[i]:+.4f} +/- {rand_delta_stds[i]:.4f}")

print("=== Scaling data (coherent) ===")
for i, s in enumerate(sizes):
    if np.isnan(coh_acc_means[i]):
        print(f"  {s.replace(chr(10),' ')}: N/A")
    else:
        print(f"  {s.replace(chr(10),' ')}: acc={coh_acc_means[i]*100:.1f}% +/- {coh_acc_stds[i]*100:.1f}%, "
              f"delta={coh_delta_means[i]:+.4f} +/- {coh_delta_stds[i]:.4f}")


# ============================================================
#  Figure 6: Scaling (two panels)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left panel: Paired accuracy bar chart ---
ax = axes[0]
x = np.arange(len(sizes))
width = 0.32

# Random bars
bars_rand = ax.bar(x - width/2, [m * 100 for m in rand_acc_means],
                   width, yerr=[s * 100 for s in rand_acc_stds],
                   capsize=5, color='#3b82f6', edgecolor='white', linewidth=1.5,
                   label='Random errors', zorder=5)

# Coherent bars (skip NaN)
coh_x = []
coh_y = []
coh_err = []
for i in range(len(sizes)):
    if not np.isnan(coh_acc_means[i]):
        coh_x.append(x[i] + width/2)
        coh_y.append(coh_acc_means[i] * 100)
        coh_err.append(coh_acc_stds[i] * 100)

ax.bar(coh_x, coh_y, width, yerr=coh_err, capsize=5,
       color='#8b5cf6', edgecolor='white', linewidth=1.5,
       label='Coherent errors', zorder=5)

# Chance level
ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance level')

# Value labels on random bars
for bar, m in zip(bars_rand, rand_acc_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{m*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#3b82f6')

# Value labels on coherent bars
for xi, yi in zip(coh_x, coh_y):
    ax.text(xi, yi + 1.5, f'{yi:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#8b5cf6')

ax.set_xlabel('Model size', fontsize=12)
ax.set_ylabel('Pair accuracy (%)', fontsize=12)
ax.set_title('Fixed-Step Size Trend', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sizes, fontsize=10)
ax.set_ylim(40, 100)
ax.legend(fontsize=10, loc='center right')
ax.grid(True, alpha=0.3, axis='y')

# --- Right panel: Paired ΔLoss line plot ---
ax = axes[1]

# Random line
rand_valid = [(sv, m, s) for sv, m, s in zip(size_values, rand_delta_means, rand_delta_stds)]
ax.errorbar([r[0] for r in rand_valid], [r[1] for r in rand_valid],
            yerr=[r[2] for r in rand_valid],
            fmt='o-', markersize=8, capsize=5, color='#3b82f6', linewidth=2,
            markeredgewidth=2, label='Random errors', zorder=5)

# Coherent line (skip NaN)
coh_valid = [(sv, m, s) for sv, m, s in zip(size_values, coh_delta_means, coh_delta_stds)
             if not np.isnan(m)]
ax.errorbar([c[0] for c in coh_valid], [c[1] for c in coh_valid],
            yerr=[c[2] for c in coh_valid],
            fmt='D-', markersize=8, capsize=5, color='#8b5cf6', linewidth=2,
            markeredgewidth=2, label='Coherent errors', zorder=5)

# Zero line
ax.axhline(y=0, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2)

ax.set_xlabel('Model size (millions of parameters)', fontsize=12)
ax.set_ylabel('Avg ΔLoss (paired)', fontsize=12)
ax.set_title('Paired ΔLoss by Model Size', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks(size_values)
ax.set_xticklabels(['3.5M', '11M', '26M', '86M'])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure6_scaling.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure6_scaling.pdf'), bbox_inches='tight')
print("Saved figure6_scaling")


# ============================================================
#  Gather multi-rule data
# ============================================================

mr2_accs, mr2_deltas = load_seeds('multirule_2_50_50_tiny_seed{seed}', filename='eval_paired_matched.json')
mr3_accs, mr3_deltas = load_seeds('multirule_3_50_50_tiny_seed{seed}', filename='eval_paired_matched.json')
mr5_accs, mr5_deltas = load_seeds('multirule_5_50_50_tiny_seed{seed}', filename='eval_paired_matched.json')
mr10_accs, mr10_deltas = load_seeds('multirule_10_50_50_tiny_seed{seed}', filename='eval_paired_matched.json')

n1_accs, _ = load_seeds('coherent_50_50_tiny_seed{seed}', filename='eval_paired_multirule_n1.json')

# N=1 is the coherent baseline evaluated on the matched N=1 multirule test.
all_n = [1, 2, 3, 5, 10]
all_acc_means = [
    np.mean(n1_accs),
    np.mean(mr2_accs),
    np.mean(mr3_accs),
    np.mean(mr5_accs),
    np.mean(mr10_accs),
]
all_acc_stds = [
    np.std(n1_accs),
    np.std(mr2_accs),
    np.std(mr3_accs),
    np.std(mr5_accs),
    np.std(mr10_accs),
]
random_acc_mean = np.mean(tiny_rand_accs)

print("\n=== Multi-rule data ===")
for n, m, s in zip(all_n, all_acc_means, all_acc_stds):
    print(f"  N={n}: acc={m*100:.1f}% +/- {s*100:.1f}%")
print(f"  N=inf (random): acc={random_acc_mean*100:.1f}%")


# ============================================================
#  Figure 7: Multi-rule
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

# X positions: categorical with labels
x_labels = ['1\n(coherent)', '2', '3', '5', '10', r'$\infty$' + '\n(random)']
x_pos = np.arange(len(x_labels))

# Accuracy values (add random as last point)
acc_means = all_acc_means + [random_acc_mean]
acc_stds = all_acc_stds + [np.std(tiny_rand_accs)]

# Color gradient from purple (N=1) through intermediate to blue (N=inf)
colors = ['#8b5cf6', '#7c6dd8', '#6d7eba', '#5e8f9c', '#4fa07e', '#3b82f6']

# Plot points with error bars
for i, (xp, ym, ys, c) in enumerate(zip(x_pos, acc_means, acc_stds, colors)):
    ax.errorbar(xp, ym * 100, yerr=ys * 100, fmt='o', markersize=12, capsize=6,
                color=c, linewidth=2, markeredgewidth=2, markeredgecolor='white', zorder=6)

# Connect with line
ax.plot(x_pos, [m * 100 for m in acc_means], '-', color='#64748b', alpha=0.6,
        linewidth=2, zorder=4)

# Chance level
ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance level (50%)')

# Random errors reference line
ax.axhline(y=random_acc_mean * 100, color='#3b82f6', linestyle='--', alpha=0.5, linewidth=1.5,
           label=f'Random errors N\u2192\u221e ({random_acc_mean*100:.1f}%)')

# Value labels
for i, (xp, ym) in enumerate(zip(x_pos, acc_means)):
    offset = -4 if i == 0 else 2.5
    ax.text(xp, ym * 100 + offset, f'{ym*100:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=colors[i])

# Annotation: steepest early rise
ax.annotate('Largest jump from N=1 to N=2',
            xy=(0.9, (acc_means[0] * 100 + acc_means[1] * 100) / 2),
            xytext=(2.0, 60),
            fontsize=10, color='#dc2626', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.8),
            ha='center')

ax.set_xlabel('Number of error rules N', fontsize=13)
ax.set_ylabel('Pair accuracy (%)', fontsize=13)
ax.set_title('Matched Multi-Rule Evaluation', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylim(40, 100)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

# Subtitle annotation
ax.text(0.5, -0.15, 'N=1: one coherent rule on a matched paired test, N\u2192\u221e: random paired benchmark',
        transform=ax.transAxes, ha='center', fontsize=10, color='#64748b', style='italic')

plt.subplots_adjust(bottom=0.18)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure7_multirule.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure7_multirule.pdf'), bbox_inches='tight')
print("Saved figure7_multirule")
