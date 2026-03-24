"""Generate conceptual figure for the MDL Proposition (Section 3.5).

Shows description length K(T) for truth vs falsehood in three regimes,
alongside the observed pair accuracy for each regime.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                 gridspec_kw={'width_ratios': [1, 1.2]})

# === Left panel: Conceptual model — description length ===

regimes = ['Coherent\nerrors\n(1 rule)', 'Multi-rule\nerrors\n(N rules)', 'Random\nerrors\n(unique)']
x = np.arange(len(regimes))
width = 0.32

# Description lengths (conceptual, not to scale — illustrative)
k_truth = [1.0, 1.0, 1.0]  # truth is always compact
k_false = [1.0, 2.5, 8.0]  # coherent ≈ truth, multi-rule > truth, random >> truth

bars_t = ax1.bar(x - width/2, k_truth, width, color='#22c55e', edgecolor='white',
                  linewidth=1.5, label='K(Truth)', zorder=5)
bars_f = ax1.bar(x + width/2, k_false, width, color='#ef4444', edgecolor='white',
                  linewidth=1.5, label='K(Falsehood)', zorder=5)

# Annotations
ax1.annotate('K(T$_1$) = K(T$_2$)\nNo preference',
            xy=(0, 1.1), ha='center', fontsize=9, color='#64748b',
            fontweight='bold')
ax1.annotate('K(T$_2$) > K(T$_1$)\nTruth preferred',
            xy=(1, 2.7), ha='center', fontsize=9, color='#64748b',
            fontweight='bold')
ax1.annotate('K(T$_2$) >> K(T$_1$)\nTruth strongly\npreferred',
            xy=(2, 8.3), ha='center', fontsize=9, color='#64748b',
            fontweight='bold')

ax1.set_ylabel('Description length K(T)', fontsize=12)
ax1.set_title('(a) MDL Prediction: Description Length', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(regimes, fontsize=10)
ax1.set_ylim(0, 10.5)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_yticks([])  # conceptual, no numeric scale

# === Right panel: Observed pair accuracy ===

# Data from experiments
labels = ['Coherent\n(1 rule)', 'Multi-rule\nN=2', 'Multi-rule\nN=3', 'Multi-rule\nN=5',
          'Multi-rule\nN=10', 'Random\n(unique)']
accuracies = [46.6, 77.6, 82.8, 84.8, 88.3, 83.1]
colors = ['#8b5cf6', '#7c6dd8', '#6d7eba', '#5e8f9c', '#4fa07e', '#3b82f6']

x2 = np.arange(len(labels))
bars = ax2.bar(x2, accuracies, 0.6, color=colors, edgecolor='white', linewidth=1.5, zorder=5)

# Chance level / baseline
ax2.axhspan(48.5, 51.5, color='#ef4444', alpha=0.06, zorder=0)
ax2.axhline(y=50, color='#ef4444', linestyle='--', alpha=0.8, linewidth=2,
            label='Chance baseline (50%)')

# Value labels
for bar, acc in zip(bars, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Annotation for steep rise
ax2.annotate('', xy=(0.1, 46.6), xytext=(0.9, 77.6),
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))
ax2.text(0.15, 60, 'Steep\nearly rise', fontsize=9, color='#dc2626', fontweight='bold')

ax2.set_ylabel('Pair accuracy (%)', fontsize=12)
ax2.set_xlabel('Dashed red line at 50% = chance baseline (random choice, no preference)',
               fontsize=10, color='#dc2626', labelpad=10)
ax2.set_title('(b) Observed Results (tiny, 3.5M)', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylim(40, 100)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure_conceptual.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure_conceptual.pdf'), bbox_inches='tight')
print("Saved figure_conceptual")
