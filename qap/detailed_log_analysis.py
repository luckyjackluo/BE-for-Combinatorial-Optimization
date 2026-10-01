#!/usr/bin/env python3
"""
Detailed analysis of each BE log file to help identify configurations.
"""

import re
from pathlib import Path

log_dir = Path('log')

# Analyze each GD log
print("="*80)
print("GD (Grad+Sink) LOGS")
print("="*80)

for log_file in sorted(log_dir.glob('*gd+k+5*.log')):
    print(f"\nFile: {log_file.name}")
    print(f"Size: {log_file.stat().st_size} bytes")
    print(f"Modified: {log_file.stat().st_mtime}")

    with open(log_file, 'r') as f:
        content = f.read()

    # Check warm start
    if 'Warm start: Enabled' in content:
        print("Warm start: YES")
    else:
        print("Warm start: NO (Random init)")

    # Count instances
    instances = re.findall(r'Processing (qap_n1000_geo_id\d+\.dat)', content)
    print(f"Instances processed: {len(instances)}")
    print(f"Unique instances: {len(set(instances))}")
    if instances:
        print(f"  {', '.join(sorted(set(instances)))}")

print("\n" + "="*80)
print("PGD (FW) LOGS")
print("="*80)

for log_file in sorted(log_dir.glob('*pgd+k+5*.log')):
    print(f"\nFile: {log_file.name}")
    print(f"Size: {log_file.stat().st_size} bytes")
    print(f"Modified: {log_file.stat().st_mtime}")

    with open(log_file, 'r') as f:
        content = f.read()

    # Check warm start
    if 'Warm start: Enabled' in content:
        print("Warm start: YES (FAQ init)")
    else:
        print("Warm start: NO (Random init)")

    # Count instances
    instances = re.findall(r'Processing (qap_n1000_geo_id\d+\.dat)', content)
    print(f"Instances processed: {len(instances)}")
    print(f"Unique instances: {len(set(instances))}")
    if instances:
        print(f"  {', '.join(sorted(set(instances)))}")

print("\n" + "="*80)
print("COMPLETED INSTANCES (all methods)")
print("="*80)

completed = [
    'qap_n1000_geo_id0.dat', 'qap_n1000_geo_id1.dat', 'qap_n1000_geo_id10.dat',
    'qap_n1000_geo_id13.dat', 'qap_n1000_geo_id14.dat', 'qap_n1000_geo_id15.dat',
    'qap_n1000_geo_id16.dat', 'qap_n1000_geo_id18.dat', 'qap_n1000_geo_id19.dat',
    'qap_n1000_geo_id2.dat', 'qap_n1000_geo_id3.dat', 'qap_n1000_geo_id4.dat',
    'qap_n1000_geo_id5.dat', 'qap_n1000_geo_id6.dat', 'qap_n1000_geo_id8.dat'
]

print(f"\n{len(completed)} instances completed by ALL methods:")
print(f"{', '.join([i.replace('qap_n1000_geo_', '').replace('.dat', '') for i in completed])}")
