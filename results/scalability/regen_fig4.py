import json
import matplotlib.pyplot as plt
import numpy as np

with open('scalability_results.json', 'r') as f:
    data = json.load(f)

corpus_data = data[0]
param_values = corpus_data['parameter_values']
timing_records = corpus_data['timing_records']

total_times = []
gen_times = []
for record in timing_records:
    total = sum(r['wall_seconds'] for r in record)
    gen = sum(r['wall_seconds'] for r in record if r['stage'] == 'generation')
    total_times.append(total)
    gen_times.append(gen)

throughputs = [n * 60 / t for n, t in zip(param_values, total_times)]

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
})

# Increase width and add manual spacing to avoid overlap
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
plt.subplots_adjust(wspace=0.45, left=0.10, right=0.95, top=0.88, bottom=0.18)

# --- Left: Time vs Corpus Size ---
ax = axes[0]
ax.plot(param_values, total_times, 'o-', color='#1f77b4', linewidth=1.5, markersize=6, label='Total')
ax.plot(param_values, gen_times, 's--', color='#ff7f0e', linewidth=1.5, markersize=5, label='Generation only')
linear_ref = [total_times[0] * (v / param_values[0]) for v in param_values]
ax.plot(param_values, linear_ref, ':', color='gray', linewidth=1.0, label='Linear reference')

ax.set_xlabel('Corpus Size (chunks)')
ax.set_ylabel('Wall-Clock Time (seconds)')
ax.set_title('(a) Scalability: Time vs Corpus Size')
ax.legend(loc='upper left', frameon=True, edgecolor='gray')
ax.grid(True, linestyle='--', alpha=0.4)

# --- Right: Throughput Stability ---
ax = axes[1]
ax.plot(param_values, throughputs, 'o-', color='#2ca02c', linewidth=1.5, markersize=6)
mean_tp = np.mean(throughputs)
ax.axhline(mean_tp, color='gray', linestyle='--', linewidth=1.0, label=f'Mean: {mean_tp:.1f} pairs/min')

ax.set_xlabel('Corpus Size (chunks)')
ax.set_ylabel('Throughput (pairs / min)')
ax.set_title('(b) Throughput Stability')
ax.legend(loc='upper left', frameon=True, edgecolor='gray')
ax.grid(True, linestyle='--', alpha=0.4)

fig.savefig('fig_scalability_corpus_size.png', dpi=300, bbox_inches='tight')
fig.savefig('fig_scalability_corpus_size.pdf', bbox_inches='tight')
print("Saved fig_scalability_corpus_size.png and .pdf")
