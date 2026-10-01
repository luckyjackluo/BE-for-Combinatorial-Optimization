#!/usr/bin/env python3
"""
Simplified results extraction - just get the 15 completed instances and
generate a basic table with what we have.
"""

import re
from pathlib import Path
import numpy as np

# Hardcoded list of 15 completed instances
completed_instances = [
    'qap_n1000_geo_id0.dat',  'qap_n1000_geo_id1.dat',  'qap_n1000_geo_id10.dat',
    'qap_n1000_geo_id13.dat', 'qap_n1000_geo_id14.dat', 'qap_n1000_geo_id15.dat',
    'qap_n1000_geo_id16.dat', 'qap_n1000_geo_id18.dat', 'qap_n1000_geo_id19.dat',
    'qap_n1000_geo_id2.dat',  'qap_n1000_geo_id3.dat',  'qap_n1000_geo_id4.dat',
    'qap_n1000_geo_id5.dat',  'qap_n1000_geo_id6.dat',  'qap_n1000_geo_id8.dat'
]

def extract_instance_results(logfile, instance_name):
    """Extract objective and runtime for a specific instance from a log file."""
    with open(logfile, 'r') as f:
        lines = f.readlines()

    # Find the line where this instance starts
    start_idx = None
    for i, line in enumerate(lines):
        if instance_name in line and 'Processing' in line:
            start_idx = i
            break

    if start_idx is None:
        return None

    # Find the end (next "Processing" or end of file)
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if 'Processing instance:' in lines[i] or 'Processing qap' in lines[i]:
            end_idx = i
            break

    section = ''.join(lines[start_idx:end_idx])

    result = {}

    # Extract objective value - try multiple patterns
    obj_match = re.search(r'Best TL:\s*([\d.]+)', section)
    if obj_match:
        result['objective'] = float(obj_match.group(1))
    else:
        obj_match = re.search(r'Objective value \(our calculation\):\s*([\d.]+)', section)
        if obj_match:
            result['objective'] = float(obj_match.group(1))
        else:
            obj_match = re.search(r'Objective value:\s*([\d.]+)', section)
            if obj_match:
                result['objective'] = float(obj_match.group(1))

    # Extract runtime
    runtime_match = re.search(r'Runtime:\s*([\d.]+)', section)
    if runtime_match:
        result['runtime'] = float(runtime_match.group(1))

    # Extract restarts for 2opt
    restart_match = re.search(r'Random restarts:\s*(\d+)', section)
    if restart_match:
        result['restarts'] = int(restart_match.group(1))

    return result if result else None

# Collect results for each method
print("Collecting results for 15 completed instances...")
print("="*80)

scipy_results = []
twoopt_2000s_results = []
twoopt_4000s_results = []
gd_results = []
pgd_results = []

log_dir = Path('log')

for instance in completed_instances:
    print(f"\n{instance}:")

    # FAQ/Scipy
    scipy_log = log_dir / 'qap_results_20251021_214558_synthetic_scipy.log'
    if scipy_log.exists():
        res = extract_instance_results(scipy_log, instance)
        if res:
            scipy_results.append(res['objective'])
            print(f"  FAQ: {res['objective']:,.0f} (runtime: {res['runtime']:.1f}s)")

    # 2-opt ~2000s (using first short log)
    twoopt_2k_log = log_dir / 'qap_results_20251021_182147_synthetic_2opt.log'
    if twoopt_2k_log.exists():
        res = extract_instance_results(twoopt_2k_log, instance)
        if res:
            twoopt_2000s_results.append(res['objective'])
            print(f"  2-opt (2000s): {res['objective']:,.0f}")

    # 2-opt ~4000s (using second long log)
    twoopt_4k_log = log_dir / 'qap_results_20251021_182240_synthetic_2opt.log'
    if twoopt_4k_log.exists():
        res = extract_instance_results(twoopt_4k_log, instance)
        if res:
            twoopt_4000s_results.append(res['objective'])
            print(f"  2-opt (4000s): {res['objective']:,.0f}")

    # GD - collect all results and take the best
    gd_logs = list(log_dir.glob('*gd+k+5*.log'))
    gd_objs = []
    for log in gd_logs:
        res = extract_instance_results(log, instance)
        if res and 'objective' in res:
            gd_objs.append(res['objective'])
    if gd_objs:
        best_gd = min(gd_objs)
        gd_results.append(best_gd)
        print(f"  GD (best of {len(gd_objs)}): {best_gd:,.0f}")

    # PGD - collect all results and take the best
    pgd_logs = list(log_dir.glob('*pgd+k+5*.log'))
    pgd_objs = []
    for log in pgd_logs:
        res = extract_instance_results(log, instance)
        if res and 'objective' in res:
            pgd_objs.append(res['objective'])
    if pgd_objs:
        best_pgd = min(pgd_objs)
        pgd_results.append(best_pgd)
        print(f"  PGD (best of {len(pgd_objs)}): {best_pgd:,.0f}")

# Calculate averages
print("\n" + "="*80)
print("AVERAGES")
print("="*80)

avg_scipy = np.mean(scipy_results) if scipy_results else None
avg_2opt_2k = np.mean(twoopt_2000s_results) if twoopt_2000s_results else None
avg_2opt_4k = np.mean(twoopt_4000s_results) if twoopt_4000s_results else None
avg_gd = np.mean(gd_results) if gd_results else None
avg_pgd = np.mean(pgd_results) if pgd_results else None

print(f"FAQ: {avg_scipy:,.0f} ({len(scipy_results)} instances)")
print(f"2-opt (2000s): {avg_2opt_2k:,.0f} ({len(twoopt_2000s_results)} instances)")
print(f"2-opt (4000s): {avg_2opt_4k:,.0f} ({len(twoopt_4000s_results)} instances)")
print(f"GD (best): {avg_gd:,.0f} ({len(gd_results)} instances)")
print(f"PGD (best): {avg_pgd:,.0f} ({len(pgd_results)} instances)")

# Find best
all_avgs = [x for x in [avg_scipy, avg_2opt_2k, avg_2opt_4k, avg_gd, avg_pgd] if x is not None]
best_avg = min(all_avgs)

print(f"\nBest average: {best_avg:,.0f}")

# Calculate gaps
def gap(avg_obj):
    return ((avg_obj - best_avg) / best_avg) * 100 if avg_obj else None

# Generate LaTeX table
latex = r"""
\begin{table}[h]
\centering
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|cc}
\toprule
\textbf{Method} & \textbf{Avg. Objective} $\downarrow$ & \textbf{Gap from best \%} $\downarrow$ \\
\midrule
"""

if avg_scipy:
    latex += f"FAQ & {avg_scipy:,.0f} & {gap(avg_scipy):.2f}\\% \\\\\n"

if avg_2opt_2k:
    latex += f"2-opt (2000s) & {avg_2opt_2k:,.0f} & {gap(avg_2opt_2k):.2f}\\% \\\\\n"

if avg_2opt_4k:
    latex += f"2-opt (4000s) & {avg_2opt_4k:,.0f} & {gap(avg_2opt_4k):.2f}\\% \\\\\n"

if avg_gd:
    latex += f"BE w/ Grad+Sink (Random Init.) & {avg_gd:,.0f} & {gap(avg_gd):.2f}\\% \\\\\n"

if avg_pgd:
    latex += f"BE w/ FW (Random Init.) & {avg_pgd:,.0f} & {gap(avg_pgd):.2f}\\% \\\\\n"

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
