#!/usr/bin/env python3
"""
Take the best result for each completed instance across all logs of the same method type.
This handles the case where experiments are still running and some logs are more complete than others.
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

# Group logs by method type
method_logs = {
    'FAQ': ['qap_results_20251021_214558_synthetic_scipy.log'],
    '2opt': ['qap_results_20251021_182147_synthetic_2opt.log',
             'qap_results_20251021_182240_synthetic_2opt.log'],
    'GD': ['results_0.01_qap_gd+k+5_synthetic_0_20251021_231101.log',
           'results_0.01_qap_gd+k+5_synthetic_0_20251021_231416.log',
           'results_0.01_qap_gd+k+5_synthetic_0_20251022_151557.log'],
    'PGD': ['results_0.01_qap_pgd+k+5_synthetic_0_20251022_091202.log',
            'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091943.log'],
    'PGD_FAQ': ['results_0.01_qap_pgd+k+5_synthetic_0_20251022_152104.log',
                'results_0.01_qap_pgd+k+5_synthetic_0_20251022_152129.log']
}

# Collect best result for each instance per method
best_results = {}

for method, log_files in method_logs.items():
    print(f"\n{'='*60}")
    print(f"{method}")
    print(f"{'='*60}")

    best_results[method] = {}

    for instance in completed_instances:
        objectives = []

        for log_file in log_files:
            log_path = log_dir / log_file
            if log_path.exists():
                obj = extract_objective(log_path, instance)
                if obj:
                    objectives.append((obj, log_file))

        if objectives:
            best_obj, best_log = min(objectives, key=lambda x: x[0])
            best_results[method][instance] = best_obj
            inst_short = instance.replace('qap_n1000_geo_', '').replace('.dat', '')
            print(f"{inst_short:5s}: {best_obj:>12,.0f}  (from {best_log})")

    if best_results[method]:
        avg = np.mean(list(best_results[method].values()))
        print(f"\nAverage: {avg:,.0f} ({len(best_results[method])} instances)")

# Generate summary table
print("\n" + "="*80)
print("SUMMARY - BEST RESULTS")
print("="*80)

method_avgs = {}
for method in ['FAQ', '2opt', 'GD', 'PGD', 'PGD_FAQ']:
    if method in best_results and best_results[method]:
        avg = np.mean(list(best_results[method].values()))
        method_avgs[method] = avg
        print(f"{method:12s}: {avg:>15,.0f}  ({len(best_results[method])} instances)")

best_avg = min(method_avgs.values())
print(f"\nBest average: {best_avg:,.0f}")

for method, avg in method_avgs.items():
    gap = ((avg - best_avg) / best_avg) * 100
    print(f"{method:12s}: Gap = {gap:6.2f}%")
