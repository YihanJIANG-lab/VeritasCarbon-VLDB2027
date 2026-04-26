"""Regenerate Fig 3: Baseline comparison 2x3 grid (ROUGE-L, BLEU-4, Distinct-2, Distinct-3, Domain Relevance, FactCheck)"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open('results/outputs/intrinsic_comparison.json') as f:
    data = json.load(f)

methods = ['CoDE (ours)', 'direct_prompting', 'self_instruct', 'wizardlm_evol']
labels = ['CoDE (ours)', 'Direct Prompting', 'Self-Instruct', 'WizardLM-Evol']
metrics = [
    ('rouge_l', 'ROUGE-L'),
    ('bleu4', 'BLEU-4'),
    ('distinct_2', 'Distinct-2'),
    ('distinct_3', 'Distinct-3'),
    ('domain_relevance', 'Domain Relevance'),
    ('factcheck', 'FactCheck'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (key, title) in enumerate(metrics):
    ax = axes[idx]
    vals = [data[m][key] for m in methods]
    bars = ax.bar(labels, vals, color='#1f77b4', width=0.6)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=30, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ax.get_ylim()[1]*0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, max(vals) * 1.15)

fig.suptitle('Intrinsic Evaluation: CoDE vs. Baselines', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/figures_and_tables/fig_baseline_comparison_6panel.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved fig_baseline_comparison_6panel.png')
