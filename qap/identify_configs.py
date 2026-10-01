#!/usr/bin/env python3
"""
Identify which log files correspond to which time budgets by comparing objectives.
4000s should have better (lower) objectives than 2000s.
FAQ warm start should have best results.
"""

import re
from pathlib import Path

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
    return None

log_dir = Path('log')

# Test instances
test_instances = ['qap_n1000_geo_id0.dat', 'qap_n1000_geo_id1.dat',
                  'qap_n1000_geo_id4.dat', 'qap_n1000_geo_id5.dat']

print("="*80)
print("COMPARING OBJECTIVES ACROSS LOG FILES")
print("="*80)

print("\n--- GD (Grad+Sink) Random Init Logs ---")
gd_logs = [
    'results_0.01_qap_gd+k+5_synthetic_0_20251021_231101.log',
    'results_0.01_qap_gd+k+5_synthetic_0_20251021_231416.log',
    'results_0.01_qap_gd+k+5_synthetic_0_20251022_151557.log'
]

for log_name in gd_logs:
    log_path = log_dir / log_name
    print(f"\n{log_name}:")
    objs = []
    for inst in test_instances:
        obj = extract_objective(log_path, inst)
        if obj:
            objs.append(obj)
            print(f"  {inst}: {obj:,.0f}")
    if objs:
        print(f"  Average: {sum(objs)/len(objs):,.0f}")

print("\n--- PGD (FW) Random Init Logs ---")
pgd_random_logs = [
    'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091202.log',
    'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091943.log'
]

for log_name in pgd_random_logs:
    log_path = log_dir / log_name
    print(f"\n{log_name}:")
    objs = []
    for inst in test_instances:
        obj = extract_objective(log_path, inst)
        if obj:
            objs.append(obj)
            print(f"  {inst}: {obj:,.0f}")
    if objs:
        print(f"  Average: {sum(objs)/len(objs):,.0f}")

print("\n--- PGD (FW) FAQ Warm Start Logs ---")
pgd_faq_logs = [
    'results_0.01_qap_pgd+k+5_synthetic_0_20251022_152104.log',
    'results_0.01_qap_pgd+k+5_synthetic_0_20251022_152129.log'
]

for log_name in pgd_faq_logs:
    log_path = log_dir / log_name
    print(f"\n{log_name}:")
    objs = []
    for inst in test_instances:
        obj = extract_objective(log_path, inst)
        if obj:
            objs.append(obj)
            print(f"  {inst}: {obj:,.0f}")
    if objs:
        print(f"  Average: {sum(objs)/len(objs):,.0f}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("Lower objectives = better performance")
print("4000s should have lower objectives than 2000s (same method)")
print("FAQ warm start should have lowest objectives overall")
