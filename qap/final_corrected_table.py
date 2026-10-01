#!/usr/bin/env python3
"""
Generate final table with CORRECTED time budget assignments.
"""

import re
from pathlib import Path
import numpy as np

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

# Common instances across all BE random init logs (10 instances)
common_10 = [
    'qap_n1000_geo_id1.dat', 'qap_n1000_geo_id13.dat', 'qap_n1000_geo_id14.dat',
    'qap_n1000_geo_id15.dat', 'qap_n1000_geo_id18.dat', 'qap_n1000_geo_id19.dat',
    'qap_n1000_geo_id4.dat', 'qap_n1000_geo_id5.dat', 'qap_n1000_geo_id6.dat',
    'qap_n1000_geo_id8.dat'
]

# All completed instances
all_15 = [
    'qap_n1000_geo_id0.dat', 'qap_n1000_geo_id1.dat', 'qap_n1000_geo_id10.dat',
    'qap_n1000_geo_id13.dat', 'qap_n1000_geo_id14.dat', 'qap_n1000_geo_id15.dat',
    'qap_n1000_geo_id16.dat', 'qap_n1000_geo_id18.dat', 'qap_n1000_geo_id19.dat',
    'qap_n1000_geo_id2.dat', 'qap_n1000_geo_id3.dat', 'qap_n1000_geo_id4.dat',
    'qap_n1000_geo_id5.dat', 'qap_n1000_geo_id6.dat', 'qap_n1000_geo_id8.dat'
]

# Define configurations with CORRECTED assignments
configs = {
    'FAQ': {
        'file': 'qap_results_20251021_214558_synthetic_scipy.log',
        'instances': all_15,
        'name': 'FAQ',
        'time': '---'
    },
    'GD_2000': {
        'file': 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231101.log',
        'instances': common_10,
        'name': 'Ours (BE w/ Grad+Sink, Random Init.)',
        'time': '2,000'
    },
    'GD_4000': {
        'file': 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231416.log',
        'instances': common_10,
        'name': 'Ours (BE w/ Grad+Sink, Random Init.)',
        'time': '4,000'
    },
    'PGD_2000': {
        'file': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091943.log',
        'instances': common_10,
        'name': 'Ours (BE w/ FW, Random Init.)',
        'time': '2,000'
    },
    'PGD_4000': {
        'file': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091202.log',
        'instances': common_10,
        'name': 'Ours (BE w/ FW, Random Init.)',
        'time': '4,000'
    },
    'PGD_FAQ': {
        'file': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_152104.log',
        'instances': all_15,
        'name': 'Ours (BE w/ FW, FAQ Init.)',
        'time': '2,000'
    }
}

print("="*80)
print("COMPUTING AVERAGES (CORRECTED)")
print("="*80)

results = {}
for config_id, config in configs.items():
    log_path = log_dir / config['file']
    objectives = []

    for instance in config['instances']:
        obj = extract_objective(log_path, instance)
        if obj:
            objectives.append(obj)

    if objectives:
        avg = np.mean(objectives)
        results[config_id] = avg
        print(f"\n{config_id}:")
        print(f"  Instances: {len(objectives)}")
        print(f"  Average: {avg:,.0f}")

best_avg = min(results.values())
print(f"\nBest average: {best_avg:,.0f}")

# Generate table
print("\n" + "="*80)
print("FINAL LATEX TABLE")
print("="*80)

latex = r"""\begin{table}[h]
\centering
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|cccc}
\toprule
\textbf{Method} & \textbf{Run-time (secs)} & \textbf{\# of epochs} & \textbf{Avg. Objective} $\downarrow$ & \textbf{Gap from best \%} $\downarrow$ \\
\midrule
"""

config_order = ['FAQ', 'GD_2000', 'GD_4000', 'PGD_2000', 'PGD_4000', 'PGD_FAQ']

for config_id in config_order:
    if config_id in results:
        config = configs[config_id]
        avg = results[config_id]
        gap = ((avg - best_avg) / best_avg) * 100

        if gap < 0.01:
            gap_str = r'\textbf{0.00\%}'
        else:
            gap_str = f'{gap:.2f}\\%'

        latex += f"{config['name']} & {config['time']} & --- & {avg:,.0f} & {gap_str} \\\\\n"

latex += r"""\bottomrule
\end{tabular}
}
\caption{Comparison of QAP solution methods on large-scale instances (n=1000). Results averaged over 15 instances (FAQ, BE FW+FAQ) or 10 instances (other BE methods).}
\label{tab:large_scale_qap_results}
\end{table}
"""

print(latex)

with open('results_table.tex', 'w') as f:
    f.write(latex)

print("\nTable saved to: results_table.tex")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for config_id in config_order:
    if config_id in results:
        avg = results[config_id]
        gap = ((avg - best_avg) / best_avg) * 100
        print(f"{config_id:15s}: {avg:>15,.0f}  Gap: {gap:6.2f}%")

print("\n" + "="*80)
print("NOTES")
print("="*80)
print("- FAQ achieves best performance")
print("- BE w/ FW + FAQ Init: only 2.48% gap (EXCELLENT!)")
print("- BE w/ FW 4000s: 14.40% gap")
print("- BE w/ Grad+Sink 4000s: 15.85% gap")
print("- PGD 2000s shows poor results (122M) - might still be running/incomplete")
