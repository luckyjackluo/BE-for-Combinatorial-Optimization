#!/usr/bin/env python3
"""
Parse log files to identify completed instances and generate results table.
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

def parse_scipy_log(filepath):
    """Parse FAQ (scipy) log file."""
    results = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Find all instances
    pattern = r'Processing instance: (qap_n\d+_geo_id\d+\.dat).*?Objective value \(our calculation\): ([\d.]+).*?Runtime: ([\d.]+) seconds'
    matches = re.findall(pattern, content, re.DOTALL)

    for instance, obj, runtime in matches:
        results[instance] = {
            'objective': float(obj),
            'runtime': float(runtime)
        }

    return results

def parse_2opt_log(filepath):
    """Parse 2-opt log file."""
    results = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_instance = None
    objective = None
    runtime = None
    num_restarts = None

    for line in lines:
        # Find instance
        if 'Processing instance:' in line:
            # Save previous instance if we have one
            if current_instance and objective is not None:
                results[current_instance] = {
                    'objective': objective,
                    'runtime': runtime,
                    'num_restarts': num_restarts
                }

            # Reset for new instance
            match = re.search(r'Processing instance: (qap_n\d+_geo_id\d+\.dat)', line)
            if match:
                current_instance = match.group(1)
                objective = None
                runtime = None
                num_restarts = None

        # Find objective value
        if 'Objective value:' in line:
            match = re.search(r'Objective value: ([\d.]+)', line)
            if match:
                objective = float(match.group(1))

        # Find runtime
        if 'Runtime:' in line and 'seconds' in line:
            match = re.search(r'Runtime: ([\d.]+) seconds', line)
            if match:
                runtime = float(match.group(1))

        # Find number of restarts
        if 'Random restarts:' in line:
            match = re.search(r'Random restarts: (\d+)', line)
            if match:
                num_restarts = int(match.group(1))

    # Don't forget the last instance
    if current_instance and objective is not None:
        results[current_instance] = {
            'objective': objective,
            'runtime': runtime,
            'num_restarts': num_restarts
        }

    return results

def parse_be_log(filepath):
    """Parse BE (gradient-based) log file."""
    results = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Find all instances with their summaries
    pattern = r'Processing (qap_n\d+_geo_id\d+\.dat).*?Summary for .*?Best TL: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    for instance, best_tl in matches:
        results[instance] = {
            'objective': float(best_tl)
        }

    return results

def main():
    log_dir = Path('log')

    # Store results by method
    methods = {
        'scipy': [],
        '2opt': [],
        'gd': [],
        'pgd': []
    }

    # Parse all log files
    for log_file in log_dir.glob('*.log'):
        filename = log_file.name

        if 'scipy' in filename:
            results = parse_scipy_log(log_file)
            methods['scipy'].append(results)
            print(f"Parsed {filename}: {len(results)} instances")

        elif '2opt' in filename:
            results = parse_2opt_log(log_file)
            methods['2opt'].append(results)
            print(f"Parsed {filename}: {len(results)} instances")

        elif 'pgd+k' in filename:  # Check pgd BEFORE gd
            results = parse_be_log(log_file)
            methods['pgd'].append(results)
            print(f"Parsed {filename}: {len(results)} instances")

        elif 'gd+k' in filename:
            results = parse_be_log(log_file)
            methods['gd'].append(results)
            print(f"Parsed {filename}: {len(results)} instances")

    # Find instances completed by all methods
    all_instances = set()
    for method_name, logs in methods.items():
        for log_results in logs:
            all_instances.update(log_results.keys())

    print(f"\nTotal unique instances found: {len(all_instances)}")
    print(f"Instances: {sorted(all_instances)}")

    # Check which instances are in each method type
    instances_by_method = {
        'scipy': set(),
        '2opt': set(),
        'gd': set(),
        'pgd': set()
    }

    for method_name, logs in methods.items():
        for log_results in logs:
            instances_by_method[method_name].update(log_results.keys())

    print(f"\nInstances by method:")
    print(f"  scipy (FAQ): {len(instances_by_method['scipy'])} instances")
    print(f"  2opt: {len(instances_by_method['2opt'])} instances")
    print(f"  gd (Grad+Sink): {len(instances_by_method['gd'])} instances")
    print(f"  pgd (FW): {len(instances_by_method['pgd'])} instances")

    # Find instances completed by all methods
    completed_instances = (instances_by_method['scipy'] &
                          instances_by_method['2opt'] &
                          instances_by_method['gd'] &
                          instances_by_method['pgd'])

    print(f"\nInstances completed by ALL methods: {len(completed_instances)}")
    print(f"Completed instances: {sorted(completed_instances)}")

    # Show which instances are missing from which methods
    all_inst_set = set(all_instances)
    for method_name in ['scipy', '2opt', 'gd', 'pgd']:
        missing = sorted(all_inst_set - instances_by_method[method_name])
        if missing:
            print(f"  Missing from {method_name}: {missing}")

    # Aggregate results for completed instances
    print("\n" + "="*80)
    print("AGGREGATING RESULTS FOR COMPLETED INSTANCES")
    print("="*80)

    # Collect all results for each method across completed instances
    results_data = {
        'scipy': [],
        '2opt_r2': [],
        '2opt_r3': [],
        '2opt_r5': [],
        'gd': [],
        'pgd': []
    }

    for instance in sorted(completed_instances):
        print(f"\nProcessing {instance}...")

        # Scipy/FAQ
        for log in methods['scipy']:
            if instance in log:
                results_data['scipy'].append({
                    'instance': instance,
                    'objective': log[instance]['objective'],
                    'runtime': log[instance]['runtime']
                })
                print(f"  FAQ: obj={log[instance]['objective']:.0f}, runtime={log[instance]['runtime']:.2f}s")

        # 2-opt - group by number of restarts
        for log in methods['2opt']:
            if instance in log:
                restarts = log[instance]['num_restarts']
                obj = log[instance]['objective']
                runtime = log[instance]['runtime']

                if restarts == 2:
                    results_data['2opt_r2'].append({
                        'instance': instance,
                        'objective': obj,
                        'runtime': runtime
                    })
                elif restarts == 3:
                    results_data['2opt_r3'].append({
                        'instance': instance,
                        'objective': obj,
                        'runtime': runtime
                    })
                elif restarts == 5:
                    results_data['2opt_r5'].append({
                        'instance': instance,
                        'objective': obj,
                        'runtime': runtime
                    })

                print(f"  2-opt (R={restarts}): obj={obj:.0f}, runtime={runtime:.2f}s")

        # GD
        for log in methods['gd']:
            if instance in log:
                results_data['gd'].append({
                    'instance': instance,
                    'objective': log[instance]['objective']
                })
                print(f"  GD: obj={log[instance]['objective']:.0f}")

        # PGD
        for log in methods['pgd']:
            if instance in log:
                results_data['pgd'].append({
                    'instance': instance,
                    'objective': log[instance]['objective']
                })
                print(f"  PGD: obj={log[instance]['objective']:.0f}")

    return results_data, completed_instances

def generate_table(results_data, completed_instances):
    """Generate LaTeX table from results."""
    print("\n" + "="*80)
    print("CALCULATING AVERAGES")
    print("="*80)

    # Calculate averages for each method
    averages = {}

    for method_name, results in results_data.items():
        if not results:
            continue

        objectives = [r['objective'] for r in results]
        avg_obj = np.mean(objectives)

        if 'runtime' in results[0]:
            runtimes = [r['runtime'] for r in results]
            avg_runtime = np.mean(runtimes)
        else:
            avg_runtime = None

        averages[method_name] = {
            'avg_objective': avg_obj,
            'avg_runtime': avg_runtime,
            'count': len(results)
        }

        print(f"{method_name}: {len(results)} instances, avg obj = {avg_obj:,.0f}", end="")
        if avg_runtime:
            print(f", avg runtime = {avg_runtime:.1f}s")
        else:
            print()

    # Find best average objective
    all_objs = [v['avg_objective'] for v in averages.values()]
    best_obj = min(all_objs)
    print(f"\nBest average objective: {best_obj:,.0f}")

    # Calculate gaps
    for method_name in averages:
        avg_obj = averages[method_name]['avg_objective']
        gap = ((avg_obj - best_obj) / best_obj) * 100
        averages[method_name]['gap'] = gap

    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)

    latex = r"""
\begin{table}[h]
\centering
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|cccc}
\toprule
\textbf{Method} & \textbf{Run-time (secs)} & \textbf{\# of epochs} & \textbf{Avg. Objective} $\downarrow$ & \textbf{Gap from best \%} $\downarrow$ \\
\midrule
"""

    # Add rows for each method
    if 'scipy' in averages:
        a = averages['scipy']
        latex += f"FAQ & {a['avg_runtime']:.0f} & --- & {a['avg_objective']:,.0f} & {a['gap']:.2f}\\% \\\\\n"

    if '2opt_r3' in averages:
        a = averages['2opt_r3']
        latex += f"2-opt (Restart=3) & {a['avg_runtime']:,.0f} & --- & {a['avg_objective']:,.0f} & {a['gap']:.2f}\\% \\\\\n"

    if '2opt_r5' in averages:
        a = averages['2opt_r5']
        latex += f"2-opt (Restart=5) & {a['avg_runtime']:,.0f} & --- & {a['avg_objective']:,.0f} & {a['gap']:.2f}\\% \\\\\n"

    if 'gd' in averages:
        a = averages['gd']
        latex += f"Ours (BE w/ Grad+Sink, Random Init.) & --- & --- & {a['avg_objective']:,.0f} & {a['gap']:.2f}\\% \\\\\n"

    if 'pgd' in averages:
        a = averages['pgd']
        latex += f"Ours (BE w/ FW, Random Init.) & --- & --- & {a['avg_objective']:,.0f} & {a['gap']:.2f}\\% \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
}
\caption{Comparison of QAP solution methods on large-scale instances (n=1000). Results averaged over """ + str(len(completed_instances)) + r""" instances.}
\label{tab:large_scale_qap_results}
\end{table}
"""

    print(latex)

    # Also save to file
    with open('results_table.tex', 'w') as f:
        f.write(latex)

    print("\nTable saved to: results_table.tex")

    return averages


if __name__ == '__main__':
    results_data, completed_instances = main()
    averages = generate_table(results_data, completed_instances)
