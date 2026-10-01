#!/usr/bin/env python3
"""
Compare GD logs to figure out which is 2000s vs 4000s.
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

gd_231101 = log_dir / 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231101.log'
gd_231416 = log_dir / 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231416.log'

print("Comparing GD logs:")
print("="*60)
print(f"{'Instance':<15} | {'231101':<15} | {'231416':<15} | Better")
print("="*60)

for inst in test_instances:
    obj_231101 = extract_objective(gd_231101, inst)
    obj_231416 = extract_objective(gd_231416, inst)

    inst_short = inst.replace('qap_n1000_geo_', '').replace('.dat', '')
    better = "231101" if obj_231101 and obj_231416 and obj_231101 < obj_231416 else "231416"

    print(f"{inst_short:<15} | {obj_231101:>13,.0f} | {obj_231416:>13,.0f} | {better}")

print("\nConclusion:")
print("If 231416 consistently has better (lower) objectives,")
print("then 231416 = 4000s and 231101 = 2000s")
