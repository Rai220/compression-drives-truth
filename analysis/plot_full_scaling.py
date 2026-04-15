"""Generate Figure 3: Cross-setup scaling summary.

Shows GPT-2 family (3.5M-86M), Qwen3-0.6B, and Qwen3 ~1B
with visual separation between different setups.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')

# ============================================================
#  Data (from paper Tables 3, 6, 8)
# ============================================================

# GPT-2 family (denoising, math-only, 2-4 seeds)
gpt2_sizes = [3.5, 12, 26, 86]
gpt2_random_acc = [65.3, 74.6, 81.1, 85.2]
gpt2_random_std = [1.3, 1.6, 1.2, 2.3]
gpt2_coherent_acc = [43.5, 44.5, 45.8, 51.0]
gpt2_coherent_std = [2.6, 3.0, 3.4, 0.8]
gpt2_seeds = [4, 4, 4, 2]

# Qwen3-0.6B (denoising, math-only, 3 seeds)
qwen_size = 420
qwen_random_acc = 86.8
qwen_random_std = 2.2  # from per-seed: 85.3, 85.7, 89.3
qwen_coherent_acc = 50.6
qwen_coherent_std = 1.9

# Qwen3 ~1B (denoising, FineWeb-Edu + 8% math, 1 seed)
qwen1b_size = 1000
qwen1b_random_acc = 76.8
qwen1b_coherent_acc = 46.7

# ============================================================
#  Plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# --- Random errors ---
# GPT-2 (solid circles, solid line)
ax.errorbar(gpt2_sizes, gpt2_random_acc, yerr=gpt2_random_std,
            fmt='o-', markersize=9, capsize=5, color='#3b82f6', linewidth=2.5,
            markeredgewidth=2, markeredgecolor='white',
            label='GPT-2 random (math-only, 2–4 seeds)', zorder=6)

# Qwen3-0.6B (square, no connecting line to GPT-2)
ax.errorbar([qwen_size], [qwen_random_acc], yerr=[qwen_random_std],
            fmt='s', markersize=11, capsize=5, color='#059669', linewidth=2,
            markeredgewidth=2, markeredgecolor='white',
            label='Qwen3-0.6B random (math-only, 3 seeds)', zorder=7)

# Qwen3 ~1B (diamond, distinct)
ax.plot([qwen1b_size], [qwen1b_random_acc],
        marker='D', markersize=11, color='#dc2626',
        markeredgewidth=2, markeredgecolor='white',
        label='Qwen3 ~1B random (FineWeb+math, 1 seed)', zorder=7)

# --- Coherent errors ---
# GPT-2 (open circles, dashed line)
ax.errorbar(gpt2_sizes, gpt2_coherent_acc, yerr=gpt2_coherent_std,
            fmt='o--', markersize=9, capsize=5, color='#3b82f6', linewidth=1.5,
            markeredgewidth=2, markerfacecolor='white',
            label='GPT-2 coherent (math-only)', zorder=5)

# Qwen3-0.6B coherent
ax.errorbar([qwen_size], [qwen_coherent_acc], yerr=[qwen_coherent_std],
            fmt='s', markersize=11, capsize=5, color='#059669', linewidth=2,
            markeredgewidth=2, markerfacecolor='white',
            zorder=6)

# Qwen3 ~1B coherent
ax.plot([qwen1b_size], [qwen1b_coherent_acc],
        marker='D', markersize=11, color='#dc2626',
        markeredgewidth=2, markerfacecolor='white',
        zorder=6)

# Chance line
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.6, linewidth=1.5, label='Chance (50%)')

# Value labels for key points
for x, y, label in [(86, 85.2, '85.2%'), (420, 86.8, '86.8%'), (1000, 76.8, '76.8%')]:
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5),
                fontsize=9, fontweight='bold', color='#1e293b')

for x, y, label in [(86, 51.0, '51.0%'), (420, 50.6, '50.6%'), (1000, 46.7, '46.7%')]:
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, -12),
                fontsize=9, color='#64748b')

# Vertical separators between setups
ax.axvline(x=200, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)
ax.axvline(x=700, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)

# Setup labels at top
ax.text(20, 97, 'GPT-2\n(math-only)', ha='center', fontsize=8, color='#64748b', style='italic')
ax.text(420, 97, 'Qwen3-0.6B\n(math-only)', ha='center', fontsize=8, color='#64748b', style='italic')
ax.text(1000, 97, 'Qwen3 ~1B\n(FineWeb+math)', ha='center', fontsize=8, color='#64748b', style='italic')

ax.set_xscale('log')
ax.set_xlabel('Parameters (millions)', fontsize=12)
ax.set_ylabel('Paired accuracy (%)', fontsize=12)
ax.set_title('Cross-Setup Scaling Summary', fontsize=14, fontweight='bold')
ax.set_xticks([3.5, 12, 26, 86, 420, 1000])
ax.set_xticklabels(['3.5M', '12M', '26M', '86M', '420M', '~1B'])
ax.set_ylim(35, 100)
ax.set_xlim(2, 1500)
ax.legend(fontsize=8.5, loc='center left', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'figure_full_scaling.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'figure_full_scaling.pdf'), bbox_inches='tight')
print("Saved figure_full_scaling")
