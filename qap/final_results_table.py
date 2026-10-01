#!/usr/bin/env python3
"""
Generate final results table with all BE configurations properly identified.
"""

import re
from pathlib import Path
import numpy as np

completed_instances = [
    'qap_n1000_geo_id0.dat', 'qap_n1000_geo_id1.dat', 'qap_n1000_geo_id10.dat',
    'qap_n1000_geo_id13.dat', 'qap_n1000_geo_id14.dat', 'qap_n1000_geo_id15.dat',
    'qap_n1000_geo_id16.dat', 'qap_n1000_geo_id18.dat', 'qap_n1000_geo_id19.dat',
    'qap_n1000_geo_id2.dat', 'qap_n1000_geo_id3.dat', 'qap_n1000_geo_id4.dat',
    'qap_n1000_geo_id5.dat', 'qap_n1000_geo_id6.dat', 'qap_n1000_geo_id8.dat'
]

def extract_objective(logfile, instance_name):
    """Extract objective for a specific instance."""
    with open(logfile, 'r') as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if instance_name in line and 'Processing' in line:
            start_idx = i
            break

    if start_idx is None:
        return None

    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if 'Processing' in lines[i]:
            end_idx = i
            break

    section = ''.join(lines[start_idx:end_idx])

    # Try multiple patterns
    obj_match = re.search(r'Best TL:\s*([\d.]+)', section)
    if obj_match:
        return float(obj_match.group(1))

    obj_match = re.search(r'Objective value \(our calculation\):\s*([\d.]+)', section)
    if obj_match:
        return float(obj_match.group(1))

    obj_match = re.search(r'Objective value:\s*([\d.]+)', section)
    if obj_match:
        return float(obj_match.group(1))

    return None

log_dir = Path('log')

# Define configurations based on our analysis
configs = {
    'FAQ': {
        'file': 'qap_results_20251021_214558_synthetic_scipy.log',
        'name': 'FAQ',
        'runtime': '---',
        'epochs': '---'
    },
    '2opt_2000': {
        'file': 'qap_results_20251021_182147_synthetic_2opt.log',
        'name': '2-opt (Restart=3)',
        'runtime': '2,000',
        'epochs': '---'
    },
    '2opt_4000': {
        'file': 'qap_results_20251021_182240_synthetic_2opt.log',
        'name': '2-opt (Restart=5)',
        'runtime': '2,000',
        'epochs': '---'
    },
    'gd_2000': {
        'file': 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231101.log',
        'name': 'Ours (BE w/ Grad+Sink, Random Init.)',
        'runtime': '2,000',
        'epochs': '---'
    },
    'gd_4000': {
        'file': 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231416.log',
        'name': 'Ours (BE w/ Grad+Sink, Random Init.)',
        'runtime': '4,000',
        'epochs': '---'
    },
    'pgd_2000': {
        'file': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091202.log',
        'name': 'Ours (BE w/ FW, Random Init.)',
        'runtime': '2,000',
        'epochs': '---'
    },
    'pgd_4000': {
        'file': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091943.log',
        'name': 'Ours (BE w/ FW, Random Init.)',
        'runtime': '4,000',
        'epochs': '---'
    },
    'pgd_faq': {
        'file': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_152104.log',
        'name': 'Ours (BE w/ FW, FAQ Init.)',
        'runtime': '2,000',
        'epochs': '---'
    }
}

# Collect results
results = {}
for config_id, config in configs.items():
    log_path = log_dir / config['file']
    objectives = []

    for instance in completed_instances:
        obj = extract_objective(log_path, instance)
        if obj:
            objectives.append(obj)

    if objectives:
        results[config_id] = {
            'objectives': objectives,
            'avg': np.mean(objectives),
            'count': len(objectives)
        }
        print(f"{config['name']:50s}: {len(objectives)} instances, avg = {np.mean(objectives):,.0f}")

# Find best average
all_avgs = [v['avg'] for v in results.values()]
best_avg = min(all_avgs)

print(f"\nBest average: {best_avg:,.0f}")

# Generate LaTeX table
latex = r"""
\begin{table}[h]
\centering
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|cccc}
\toprule
\textbf{Method} & \textbf{Run-time (secs)} & \textbf{\# of epochs} & \textbf{Avg. Objective} $\downarrow$ & \textbf{Gap from best \%} $\downarrow$ \\
\midrule
"""

# Add each row in order
config_order = ['FAQ', '2opt_2000', '2opt_4000', 'gd_2000', 'gd_4000', 'pgd_2000', 'pgd_4000', 'pgd_faq']

for config_id in config_order:
    if config_id in results:
        config = configs[config_id]
        avg = results[config_id]['avg']
        gap = ((avg - best_avg) / best_avg) * 100

        # Mark the best with bold
        if abs(gap) < 0.01:
            gap_str = r'\textbf{0.00\%}'
        else:
            gap_str = f'{gap:.2f}\\%'

        latex += f"{config['name']} & {config['runtime']} & {config['epochs']} & {avg:,.0f} & {gap_str} \\\\\n"

latex += r"""
\bottomrule
\end{tabular}
}
\caption{Comparison of QAP solution methods on large-scale instances (n=1000). Results averaged over """ + str(len(completed_instances)) + r""" instances.}
\label{tab:large_scale_qap_results}
\end{table}
"""

print("\n" + "="*80)
print("LATEX TABLE")
print("="*80)
print(latex)

# Save to file
with open('results_table.tex', 'w') as f:
    f.write(latex)

print("\nTable saved to: results_table.tex")

# Also print a summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for config_id in config_order:
    if config_id in results:
        config = configs[config_id]
        avg = results[config_id]['avg']
        gap = ((avg - best_avg) / best_avg) * 100
        print(f"{config['name']:50s} | {avg:>15,.0f} | Gap: {gap:6.2f}%")
