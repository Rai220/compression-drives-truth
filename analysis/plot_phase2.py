"""Generate Figure 4: Phase 2 — Effect of observation ratio on truth bias for coherent errors."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data: Observed experiments (coherent errors, 50/50 correct/incorrect) ---
# observation_ratio -> list of (correct_loss, incorrect_loss) per seed

data_observed = {
    0: [
        (0.1411, 0.1413),
        (0.1404, 0.1410),
        (0.1424, 0.1426),
        (0.1416, 0.1424),
    ],
    10: [
        (0.1406, 0.1404),
        (0.1418, 0.1420),
        (0.1409, 0.1412),
        (0.1430, 0.1434),
    ],
    25: [
        (0.1425, 0.1429),
        (0.1436, 0.1439),
        (0.1436, 0.1440),
        (0.1443, 0.1445),
    ],
    50: [
        (0.1458, 0.1469),
        (0.1460, 0.1469),
        (0.1485, 0.1498),
        (0.1480, 0.1480),
    ],
    # 100% excluded — loss exploded (~0.32), unreliable
}

# Reference: random errors 50/50 (from Phase 1)
random_50_50_delta = 0.0115

# === Figure 4: Two-panel ===

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left panel: ΔLoss vs observation ratio ---
ax = axes[0]

obs_ratios = [0, 10, 25, 50]
mean_deltas = []
std_deltas = []

for obs in obs_ratios:
    deltas = [inc - cor for cor, inc in data_observed[obs]]
    mean_deltas.append(np.mean(deltas))
    std_deltas.append(np.std(deltas))

ax.errorbar(obs_ratios, mean_deltas, yerr=std_deltas, fmt='o-', markersize=10,
            capsize=6, color='#2563eb', linewidth=2, markeredgewidth=2, zorder=5,
            label='Coherent errors + observations')

# Reference line: random errors 50/50
ax.axhline(y=random_50_50_delta, color='#16a34a', linestyle='--', alpha=0.7,
           linewidth=2, label=f'Random errors 50/50 (ΔLoss = +{random_50_50_delta:.4f})')

# Zero line
ax.axhline(y=0, color='#ef4444', linestyle=':', alpha=0.8, linewidth=2, label='No bias (Δ = 0)')

# Individual seeds
for obs in obs_ratios:
    deltas = [inc - cor for cor, inc in data_observed[obs]]
    x_jitter = [obs + np.random.uniform(-1.5, 1.5) for _ in deltas]
    ax.scatter(x_jitter, deltas, s=30, color='#2563eb', alpha=0.4, zorder=4)

# Shade
ax.axhspan(0, 0.013, alpha=0.04, color='green')
ax.axhspan(-0.001, 0, alpha=0.04, color='red')

ax.set_xlabel('Observation ratio (%)', fontsize=12)
ax.set_ylabel('ΔLoss (incorrect − correct)', fontsize=12)
ax.set_title('Phase 2: Effect of Observations\non Truth Bias (Coherent Errors)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_xticks([0, 10, 25, 50])
ax.set_xlim(-5, 55)
ax.grid(True, alpha=0.3)

# Annotate
ax.annotate('Observations do not\nrestore strong truth bias',
            xy=(25, 0.0004), fontsize=9, color='#64748b', ha='center',
            xytext=(35, 0.0035),
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.2))

ax.annotate('Random errors\n(10× stronger)',
            xy=(48, random_50_50_delta), fontsize=8, color='#16a34a', ha='right',
            va='bottom')

# --- Right panel: Absolute losses ---
ax = axes[1]

for obs in obs_ratios:
    cor_losses = [c for c, i in data_observed[obs]]
    inc_losses = [i for c, i in data_observed[obs]]

    ax.errorbar(obs - 0.8, np.mean(cor_losses), yerr=np.std(cor_losses),
                fmt='o', markersize=8, capsize=5, color='#16a34a', linewidth=2,
                markeredgewidth=2, label='Correct test set' if obs == 0 else '')
    ax.errorbar(obs + 0.8, np.mean(inc_losses), yerr=np.std(inc_losses),
                fmt='^', markersize=8, capsize=5, color='#dc2626', linewidth=2,
                markeredgewidth=2, label='Incorrect test set' if obs == 0 else '')

# Connect means
cor_means = [np.mean([c for c, i in data_observed[obs]]) for obs in obs_ratios]
inc_means = [np.mean([i for c, i in data_observed[obs]]) for obs in obs_ratios]
ax.plot(obs_ratios, cor_means, '--', color='#16a34a', alpha=0.4, linewidth=1.5)
ax.plot(obs_ratios, inc_means, '--', color='#dc2626', alpha=0.4, linewidth=1.5)

ax.set_xlabel('Observation ratio (%)', fontsize=12)
ax.set_ylabel('Cross-entropy loss', fontsize=12)
ax.set_title('Absolute Loss: Correct vs Incorrect\n(Coherent Errors + Observations)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xticks([0, 10, 25, 50])
ax.set_xlim(-5, 55)
ax.grid(True, alpha=0.3)

# Annotate rising loss
ax.annotate('Loss rises with obs ratio\n(longer sequences)',
            xy=(50, 0.148), fontsize=8, color='#64748b', ha='center',
            xytext=(35, 0.1505),
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.2))

plt.tight_layout()
plt.savefig('results/figure4_observations.png', dpi=200, bbox_inches='tight')
plt.savefig('results/figure4_observations.pdf', bbox_inches='tight')
print("Saved figure4_observations")
