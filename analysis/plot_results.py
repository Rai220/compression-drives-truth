"""Generate figures for the paper — all experiments including coherent and contradictory."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data: Main experiment (random errors) ---

proportions = [0.10, 0.20, 0.30, 0.40, 0.50]
proportion_labels = ["10/90", "20/80", "30/70", "40/60", "50/50"]

# Per-seed results: (correct_loss, incorrect_loss)
data = {
    0.50: [
        (0.1396, 0.1508),
        (0.1383, 0.1498),
        (0.1373, 0.1489),
        (0.1384, 0.1500),
    ],
    0.40: [
        (0.1410, 0.1496),
        (0.1396, 0.1490),
        (0.1402, 0.1489),
        (0.1405, 0.1493),
    ],
    0.30: [
        (0.1432, 0.1491),
        (0.1413, 0.1485),
        (0.1427, 0.1489),
        (0.1417, 0.1481),
    ],
    0.20: [
        (0.1463, 0.1494),
        (0.1453, 0.1483),
        (0.1447, 0.1480),
        (0.1456, 0.1491),
    ],
    0.10: [
        (0.1505, 0.1488),
        (0.1504, 0.1488),
        (0.1506, 0.1486),
        (0.1499, 0.1486),
    ],
}

baseline_correct_loss = 0.1313
baseline_incorrect_loss = 0.2028

# --- Data: Coherent errors (50/50) ---
data_coherent = [
    (0.1369, 0.1361),
    (0.1371, 0.1366),
    (0.1377, 0.1376),
    (0.1379, 0.1379),
]

# --- Data: Contradictory errors (50/50) ---
data_contradictory = [
    (0.1406, 0.1411),
    (0.1413, 0.1417),
    (0.1412, 0.1417),
    (0.1394, 0.1400),
]


# === Figure 1: Two-panel main figure ===

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left panel: Delta vs proportion ---
ax = axes[0]

mean_deltas = []
for prop in proportions:
    deltas = [inc - cor for cor, inc in data[prop]]
    mean_d = np.mean(deltas)
    std_d = np.std(deltas)
    mean_deltas.append(mean_d)

    color = '#2563eb' if mean_d > 0 else '#dc2626'
    ax.errorbar(prop, mean_d, yerr=std_d, fmt='o', markersize=10, capsize=6,
                color=color, linewidth=2, markeredgewidth=2, zorder=5)

# Baseline
ax.plot(1.0, baseline_incorrect_loss - baseline_correct_loss, 's', markersize=10,
        color='#9333ea', label='Baseline (100% correct)', zorder=5)

# Connect with line
props_all = proportions + [1.0]
deltas_all = mean_deltas + [baseline_incorrect_loss - baseline_correct_loss]
ax.plot(props_all, deltas_all, '--', color='#64748b', alpha=0.5, linewidth=1.5)

# Zero line
ax.axhline(y=0, color='#ef4444', linestyle=':', alpha=0.8, linewidth=2, label='No bias (Δ = 0)')

# Shade regions
ax.axhspan(0, 0.08, alpha=0.04, color='green')
ax.axhspan(-0.005, 0, alpha=0.04, color='red')

ax.set_xlabel('Fraction of correct examples in training corpus', fontsize=12)
ax.set_ylabel('ΔLoss (incorrect − correct)', fontsize=12)
ax.set_title('Truth Bias Across Training Proportions', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0.03, 1.1)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '100%'])
ax.grid(True, alpha=0.3)

# Annotations
ax.annotate('Truth bias\n(model prefers correct)',
            xy=(0.35, 0.008), fontsize=9, color='#16a34a', ha='center', style='italic')
ax.annotate('Frequency bias\n(model prefers majority)',
            xy=(0.12, -0.003), fontsize=9, color='#dc2626', ha='center', style='italic')
ax.annotate('Crossover\n≈15%', xy=(0.155, 0.0005), fontsize=8, color='#64748b',
            ha='center', arrowprops=dict(arrowstyle='->', color='#64748b'),
            xytext=(0.22, -0.003))

# --- Right panel: Absolute losses ---
ax = axes[1]

for prop in proportions:
    cor_losses = [c for c, i in data[prop]]
    inc_losses = [i for c, i in data[prop]]
    x_jitter_cor = prop - 0.012
    x_jitter_inc = prop + 0.012

    ax.errorbar(x_jitter_cor, np.mean(cor_losses), yerr=np.std(cor_losses),
                fmt='o', markersize=8, capsize=5, color='#16a34a', linewidth=2,
                markeredgewidth=2, label='Correct test set' if prop == proportions[0] else '')
    ax.errorbar(x_jitter_inc, np.mean(inc_losses), yerr=np.std(inc_losses),
                fmt='^', markersize=8, capsize=5, color='#dc2626', linewidth=2,
                markeredgewidth=2, label='Incorrect test set' if prop == proportions[0] else '')

# Connect means with lines
cor_means = [np.mean([c for c, i in data[p]]) for p in proportions]
inc_means = [np.mean([i for c, i in data[p]]) for p in proportions]
ax.plot(proportions, cor_means, '--', color='#16a34a', alpha=0.4, linewidth=1.5)
ax.plot(proportions, inc_means, '--', color='#dc2626', alpha=0.4, linewidth=1.5)

# Baseline
ax.plot(0.985, baseline_correct_loss, 'o', markersize=8, color='#16a34a', alpha=0.4)
ax.plot(1.015, baseline_incorrect_loss, '^', markersize=8, color='#dc2626', alpha=0.4)

ax.set_xlabel('Fraction of correct examples in training corpus', fontsize=12)
ax.set_ylabel('Cross-entropy loss', fontsize=12)
ax.set_title('Loss on Correct vs Incorrect Test Sets', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0.03, 1.1)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '100%'])
ax.grid(True, alpha=0.3)

# Annotate crossover
ax.annotate('Lines cross:\ncorrect loss > incorrect',
            xy=(0.10, 0.1505), fontsize=8, color='#64748b',
            xytext=(0.18, 0.153),
            arrowprops=dict(arrowstyle='->', color='#64748b'))

plt.tight_layout()
plt.savefig('results/figure1_truth_bias.png', dpi=200, bbox_inches='tight')
plt.savefig('results/figure1_truth_bias.pdf', bbox_inches='tight')
print("Saved figure1")


# === Figure 2: Scatter plot (all experiments) ===

fig, ax = plt.subplots(figsize=(8, 6))

# Main experiment (random errors) — by proportion
colors_map = {0.10: '#ef4444', 0.20: '#f97316', 0.30: '#eab308', 0.40: '#22c55e', 0.50: '#3b82f6'}
for prop in proportions:
    cor_losses = [c for c, i in data[prop]]
    inc_losses = [i for c, i in data[prop]]
    ax.scatter(cor_losses, inc_losses, s=80, color=colors_map[prop],
               label=f'Random {int(prop*100)}/{int((1-prop)*100)}', zorder=5,
               edgecolors='white', linewidth=1)

# Coherent errors
cor_coh = [c for c, i in data_coherent]
inc_coh = [i for c, i in data_coherent]
ax.scatter(cor_coh, inc_coh, s=120, color='#8b5cf6', marker='D',
           label='Coherent 50/50', zorder=6, edgecolors='white', linewidth=1.5)

# Contradictory errors
cor_con = [c for c, i in data_contradictory]
inc_con = [i for c, i in data_contradictory]
ax.scatter(cor_con, inc_con, s=120, color='#06b6d4', marker='s',
           label='Contradictory 50/50', zorder=6, edgecolors='white', linewidth=1.5)

# Diagonal
lims = [0.132, 0.155]
ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1, label='Equal loss')
ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.05, color='green')
ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.05, color='red')

ax.set_xlabel('Loss on correct test set', fontsize=12)
ax.set_ylabel('Loss on incorrect test set', fontsize=12)
ax.set_title('All Seeds: Correct vs Incorrect Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax.annotate('Truth bias ↑', xy=(0.136, 0.151), fontsize=9, color='#16a34a', style='italic')
ax.annotate('Frequency bias ↓', xy=(0.148, 0.1355), fontsize=9, color='#dc2626', style='italic')

plt.tight_layout()
plt.savefig('results/figure2_scatter.png', dpi=200, bbox_inches='tight')
plt.savefig('results/figure2_scatter.pdf', bbox_inches='tight')
print("Saved figure2")


# === Figure 3: Error coherence spectrum (bar chart) ===

fig, ax = plt.subplots(figsize=(8, 5))

# Compute deltas for each error type
random_deltas = [inc - cor for cor, inc in data[0.50]]
contradictory_deltas = [inc - cor for cor, inc in data_contradictory]
coherent_deltas = [inc - cor for cor, inc in data_coherent]

labels = ['Random\nerrors', 'Contradictory\nerrors', 'Coherent\nerrors']
means = [np.mean(random_deltas), np.mean(contradictory_deltas), np.mean(coherent_deltas)]
stds = [np.std(random_deltas), np.std(contradictory_deltas), np.std(coherent_deltas)]
colors = ['#3b82f6', '#06b6d4', '#8b5cf6']

bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors,
              edgecolor='white', linewidth=2, width=0.55, zorder=5)

# Zero line
ax.axhline(y=0, color='#ef4444', linestyle=':', alpha=0.8, linewidth=2)

# Value labels on bars
for bar, m, s in zip(bars, means, stds):
    sign = '+' if m > 0 else ''
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.0005,
            f'{sign}{m:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Individual seeds as dots
for i, (deltas, color) in enumerate(zip([random_deltas, contradictory_deltas, coherent_deltas], colors)):
    x_jitter = np.array([i - 0.08, i + 0.08, i - 0.08, i + 0.08])
    ax.scatter(x_jitter, deltas, s=40, color='black', alpha=0.4, zorder=6)

# Annotations
ax.annotate('Strong truth bias\n(each error unique,\nhigh incompressibility)',
            xy=(0, means[0]), fontsize=8, color='#64748b', ha='center',
            xytext=(0.7, means[0] + 0.004),
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.2))

ax.annotate('No truth bias\n(consistent rules,\nequal compressibility)',
            xy=(2, means[2]), fontsize=8, color='#64748b', ha='center',
            xytext=(1.3, means[2] - 0.004),
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.2))

ax.set_ylabel('ΔLoss (incorrect − correct)', fontsize=12)
ax.set_title('Error Coherence Spectrum: Truth Bias by Error Type\n(all at 50/50 correct/incorrect ratio)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Shade
ax.axhspan(0, 0.02, alpha=0.04, color='green')
ax.axhspan(-0.003, 0, alpha=0.04, color='red')
ax.text(2.35, 0.001, 'Truth bias', fontsize=8, color='#16a34a', style='italic', va='bottom')
ax.text(2.35, -0.0005, 'Anti-bias', fontsize=8, color='#dc2626', style='italic', va='top')

plt.tight_layout()
plt.savefig('results/figure3_coherence_spectrum.png', dpi=200, bbox_inches='tight')
plt.savefig('results/figure3_coherence_spectrum.pdf', bbox_inches='tight')
print("Saved figure3")
