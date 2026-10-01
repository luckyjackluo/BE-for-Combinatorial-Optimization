#!/usr/bin/env python3
"""
Compare PGD logs to figure out which is 2000s vs 4000s.
"""

import re
from pathlib import Path

def extract_objective(logfile, instance_name):
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

test_instances = ['qap_n1000_geo_id1.dat', 'qap_n1000_geo_id4.dat',
                  'qap_n1000_geo_id5.dat', 'qap_n1000_geo_id13.dat']

pgd_091202 = log_dir / 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091202.log'
pgd_091943 = log_dir / 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091943.log'

print("Comparing PGD logs:")
print("="*60)
print(f"{'Instance':<15} | {'091202':<15} | {'091943':<15} | Better")
print("="*60)

for inst in test_instances:
    obj_091202 = extract_objective(pgd_091202, inst)
    obj_091943 = extract_objective(pgd_091943, inst)

    inst_short = inst.replace('qap_n1000_geo_', '').replace('.dat', '')
    better = "091202" if obj_091202 and obj_091943 and obj_091202 < obj_091943 else "091943"

    print(f"{inst_short:<15} | {obj_091202:>13,.0f} | {obj_091943:>13,.0f} | {better}")

print("\nConclusion:")
print("If 091202 consistently has better (lower) objectives,")
print("then 091202 = 4000s and 091943 = 2000s")
print("OR 091943 might still be running/incomplete")
