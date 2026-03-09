"""Generate Figure 9: Multi-alternative errors in the synthetic world."""

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


def load_seeds(pattern, seeds=(42, 43, 44, 45)):
    accs, deltas = [], []
    for s in seeds:
        p = os.path.join(RESULTS, pattern.format(seed=s), 'eval_paired.json')
        if os.path.exists(p):
            a, d = load_paired(p)
            accs.append(a)
            deltas.append(d)
    return accs, deltas


def load_seeds_random(pattern, seeds=(42, 43, 44, 45)):
    """Load eval_paired_random.json (comparison vs random errors)."""
    accs, deltas = [], []
    for s in seeds:
        p = os.path.join(RESULTS, pattern.format(seed=s), 'eval_paired_random.json')
        if os.path.exists(p):
            a, d = load_paired(p)
            accs.append(a)
            deltas.append(d)
    return accs, deltas


# ============================================================
#  Gather data
# ============================================================

# N=1 (coherent)
coh_accs, coh_deltas = load_seeds('world_coherent_50_50_tiny_seed{seed}')
# N=inf (random)
rand_accs, rand_deltas = load_seeds('world_random_50_50_tiny_seed{seed}')

# Multi-alt: N=2,4,8,16
ma2_accs, ma2_deltas = load_seeds('world_multialt2_50_50_tiny_seed{seed}')
ma4_accs, ma4_deltas = load_seeds('world_multialt4_50_50_tiny_seed{seed}')
ma8_accs, ma8_deltas = load_seeds('world_multialt8_50_50_tiny_seed{seed}')
ma16_accs, ma16_deltas = load_seeds('world_multialt16_50_50_tiny_seed{seed}')

# vs random comparison
ma2_vr_accs, _ = load_seeds_random('world_multialt2_50_50_tiny_seed{seed}')
ma4_vr_accs, _ = load_seeds_random('world_multialt4_50_50_tiny_seed{seed}')
ma8_vr_accs, _ = load_seeds_random('world_multialt8_50_50_tiny_seed{seed}')
ma16_vr_accs, _ = load_seeds_random('world_multialt16_50_50_tiny_seed{seed}')

# Print data
all_n = [1, 2, 4, 8, 16]
all_accs = [coh_accs, ma2_accs, ma4_accs, ma8_accs, ma16_accs]
all_vr_accs = [rand_accs, ma2_vr_accs, ma4_vr_accs, ma8_vr_accs, ma16_vr_accs]

print("=== Multi-alt world data ===")
for n, a, vr in zip(all_n, all_accs, all_vr_accs):
    print(f"  N={n}: acc vs multi-alt = {np.mean(a)*100:.1f}% +/- {np.std(a)*100:.1f}%, "
          f"acc vs random = {np.mean(vr)*100:.1f}% +/- {np.std(vr)*100:.1f}%")
print(f"  N=inf (random): acc = {np.mean(rand_accs)*100:.1f}%")

# Also load math multi-rule for comparison
math_mr_accs = {}
for n in [1, 2, 3, 5, 10]:
    if n == 1:
        accs, _ = load_seeds('coherent_50_50_tiny_seed{seed}')
    else:
        accs, _ = load_seeds(f'multirule_{n}_50_50_tiny_seed{{seed}}')
    math_mr_accs[n] = np.mean(accs) if accs else None

# Math random baseline
math_rand_accs = []
for path in ['mixed_50_50_tiny/eval_paired.json',
             'mixed_50_50_tiny_seed43/eval_paired.json',
             'mixed_50_50_tiny_seed44/eval_paired.json',
             'mixed_50_50_tiny_seed45/eval_paired.json']:
    p = os.path.join(RESULTS, path)
    if os.path.exists(p):
        a, _ = load_paired(p)
        math_rand_accs.append(a)

math_rand_mean = np.mean(math_rand_accs) if math_rand_accs else 0.83


# ============================================================
#  Figure 9: Two panels
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: Natural language (accuracy vs multi-alt, the main result) ---
ax = axes[0]

x_labels = ['1\n(coherent)', '2', '4', '8', '16', r'$\infty$' + '\n(random)']
x_pos = np.arange(len(x_labels))

# Accuracy vs own alternatives
acc_means = [np.mean(a) for a in all_accs] + [np.mean(rand_accs)]
acc_stds = [np.std(a) for a in all_accs] + [np.std(rand_accs)]

# Color gradient
colors_nl = ['#8b5cf6', '#9b59b6', '#a855f7', '#c084fc', '#d8b4fe', '#3b82f6']

for i, (xp, ym, ys, c) in enumerate(zip(x_pos, acc_means, acc_stds, colors_nl)):
    ax.errorbar(xp, ym * 100, yerr=ys * 100, fmt='o', markersize=12, capsize=6,
                color=c, linewidth=2, markeredgewidth=2, markeredgecolor='white', zorder=6)

# Connect with line
ax.plot(x_pos, [m * 100 for m in acc_means], '-', color='#64748b', alpha=0.6,
        linewidth=2, zorder=4)

# Chance level
ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance (50%)')

# Random baseline
ax.axhline(y=np.mean(rand_accs) * 100, color='#3b82f6', linestyle=':', alpha=0.5, linewidth=1.5,
           label=f'Random baseline ({np.mean(rand_accs)*100:.1f}%)')

# Value labels
for i, (xp, ym) in enumerate(zip(x_pos, acc_means)):
    offset = -4 if ym < 0.5 else 2.5
    ax.text(xp, ym * 100 + offset, f'{ym*100:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=colors_nl[i])

ax.set_xlabel('Number of alternative conclusions N', fontsize=12)
ax.set_ylabel('Pair accuracy (%)', fontsize=12)
ax.set_title('Synthetic World (Natural Language)', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylim(30, 75)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Annotation: gradual, no phase transition
ax.annotate('Gradual rise\n(no phase transition)',
            xy=(3, np.mean(ma8_accs) * 100),
            xytext=(3.5, 68),
            fontsize=10, color='#dc2626', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
            ha='center')


# --- Right panel: Comparison with math domain ---
ax = axes[1]

# Math multi-rule data
math_ns = [1, 2, 3, 5, 10]
math_accs_plot = [math_mr_accs[n] * 100 for n in math_ns if math_mr_accs[n] is not None]
math_ns_plot = [n for n in math_ns if math_mr_accs[n] is not None]

# Natural language data (N=1,2,4,8,16)
nl_ns = [1, 2, 4, 8, 16]
nl_accs_plot = [np.mean(a) * 100 for a in all_accs]

# Plot math
ax.plot(math_ns_plot, math_accs_plot, 'o-', markersize=10, color='#3b82f6',
        linewidth=2, markeredgewidth=2, markeredgecolor='white', label='Math domain', zorder=6)

# Plot natural language
ax.plot(nl_ns, nl_accs_plot, 's-', markersize=10, color='#8b5cf6',
        linewidth=2, markeredgewidth=2, markeredgecolor='white', label='Natural language', zorder=6)

# Add random baselines
ax.axhline(y=math_rand_mean * 100, color='#3b82f6', linestyle=':', alpha=0.4, linewidth=1.5)
ax.axhline(y=np.mean(rand_accs) * 100, color='#8b5cf6', linestyle=':', alpha=0.4, linewidth=1.5)

# Chance
ax.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2, label='Chance (50%)')

# Value labels for math
for n, a in zip(math_ns_plot, math_accs_plot):
    ax.text(n, a + 2, f'{a:.0f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#3b82f6')

# Value labels for NL
for n, a in zip(nl_ns, nl_accs_plot):
    offset = -4 if a < 50 else 2
    ax.text(n, a + offset, f'{a:.0f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#8b5cf6')

# Annotation
ax.annotate('Sharp jump\nin math',
            xy=(1.5, 70), xytext=(4, 72),
            fontsize=10, color='#3b82f6', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#3b82f6', lw=1.5),
            ha='center')

ax.set_xlabel('Number of error rules/alternatives N', fontsize=12)
ax.set_ylabel('Pair accuracy (%)', fontsize=12)
ax.set_title('Math vs Natural Language: Phase Transition', fontsize=13, fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xticklabels(['1', '2', '4', '8', '16'])
ax.set_ylim(30, 100)
ax.legend(fontsize=9, loc='center right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure9_world_multialt.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure9_world_multialt.pdf'), bbox_inches='tight')
print("Saved figure9_world_multialt")
