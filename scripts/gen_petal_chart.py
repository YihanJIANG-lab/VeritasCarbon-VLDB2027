"""Generate a petal/rose chart (Nightingale chart) for baseline comparison."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

with open('results/outputs/intrinsic_comparison.json') as f:
    data = json.load(f)

methods = ['CoDE (ours)', 'direct_prompting', 'self_instruct', 'wizardlm_evol']
labels  = ['CoDE (ours)', 'Direct\nPrompting', 'Self-\nInstruct', 'WizardLM-\nEvol']
metrics = ['rouge_l', 'bleu4', 'distinct_2', 'distinct_3', 'domain_relevance', 'factcheck']
metric_labels = ['ROUGE-L', 'BLEU-4', 'Distinct-2', 'Distinct-3', 'Domain\nRelevance', 'FactCheck']

# Normalize each metric to [0, 1] based on max across methods
raw = {m: [data[m][k] for k in metrics] for m in methods}
max_vals = [max(raw[m][i] for m in methods) for i in range(len(metrics))]
norm = {m: [raw[m][i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(len(metrics))] for m in methods}

n_metrics = len(metrics)
n_methods = len(methods)

# Color palettes – use distinct hues similar to the petal image (reds/pinks + blues)
colors = ['#1a5276', '#2e86c1', '#e74c3c', '#f1948a']

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, polar=True)
ax.set_position([0.0, 0.05, 0.78, 0.90])  # shift chart left, leave room for legend

# Each metric gets a sector; within each sector, methods are sub-petals
sector_width = 2 * np.pi / n_metrics
gap_ratio = 0.12  # gap between sectors
petal_width = sector_width * (1 - gap_ratio) / n_methods

import matplotlib.patches as mpatches

# Draw grouping background arc and boundary lines for each metric sector
for i in range(n_metrics):
    theta_start = i * sector_width + sector_width * gap_ratio / 2
    theta_end   = theta_start + sector_width * (1 - gap_ratio)
    theta_mid   = (theta_start + theta_end) / 2

    # Shaded background wedge (very light) to visually group the 4 petals
    n_arc = 60
    arc_thetas = np.linspace(theta_start - 0.005, theta_end + 0.005, n_arc)
    r_inner = np.full(n_arc, 0.04)
    r_outer = np.full(n_arc, 1.10)
    ax.fill_between(arc_thetas, r_inner, r_outer,
                    alpha=0.06, color='#7f8c8d', zorder=0)

    # Left boundary line
    ax.plot([theta_start - 0.005, theta_start - 0.005], [0, 1.10],
            color='#95a5a6', linewidth=1.2, linestyle='--', zorder=1)
    # Right boundary line
    ax.plot([theta_end + 0.005, theta_end + 0.005], [0, 1.10],
            color='#95a5a6', linewidth=1.2, linestyle='--', zorder=1)

for j, method in enumerate(methods):
    for i in range(n_metrics):
        theta = i * sector_width + j * petal_width + sector_width * gap_ratio / 2
        r = norm[method][i]
        bar = ax.bar(
            theta + petal_width / 2,
            r,
            width=petal_width * 0.92,
            bottom=0.05,
            color=colors[j],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.8,
        )

# Concentric grid circles with percentage labels
ax.set_ylim(0, 1.35)
circle_vals = [0.25, 0.50, 0.75, 1.0]
ax.set_rticks([v + 0.05 for v in circle_vals])
ax.set_yticklabels([f'{int(v*100)}%' for v in circle_vals], fontsize=13, color='#666')

# Metric labels at sector centers
for i, label in enumerate(metric_labels):
    angle = i * sector_width + sector_width / 2
    # Place label outside the chart
    ax.text(angle, 1.22, label, ha='center', va='center', fontsize=16,
            fontweight='bold', color='#2c3e50')

# Add value annotations on the CoDE petals (the dominant ones)
for i in range(n_metrics):
    theta = i * sector_width + 0 * petal_width + sector_width * gap_ratio / 2 + petal_width / 2
    r = norm[methods[0]][i]
    val = raw[methods[0]][i]
    ax.text(theta, r + 0.12, f'{val:.4f}', ha='center', va='center',
            fontsize=12, color=colors[0], fontweight='bold')

# Style
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids([])  # remove default angle labels
ax.spines['polar'].set_visible(False)
ax.grid(True, color='#ddd', linewidth=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[j], alpha=0.85, label=labels[j].replace('\n', ' '))
                   for j in range(n_methods)]
leg = ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.18, 0.08),
                fontsize=15, frameon=True, fancybox=True, shadow=True,
                title='Methods', title_fontsize=16)

out = 'results/figures_and_tables/fig3_petal_baseline.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
