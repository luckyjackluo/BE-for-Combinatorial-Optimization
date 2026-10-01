#!/usr/bin/env python3
"""
Check which of the 15 completed instances are in each log file.
"""

import re
from pathlib import Path

completed_instances = [
    'qap_n1000_geo_id0.dat', 'qap_n1000_geo_id1.dat', 'qap_n1000_geo_id10.dat',
    'qap_n1000_geo_id13.dat', 'qap_n1000_geo_id14.dat', 'qap_n1000_geo_id15.dat',
    'qap_n1000_geo_id16.dat', 'qap_n1000_geo_id18.dat', 'qap_n1000_geo_id19.dat',
    'qap_n1000_geo_id2.dat', 'qap_n1000_geo_id3.dat', 'qap_n1000_geo_id4.dat',
    'qap_n1000_geo_id5.dat', 'qap_n1000_geo_id6.dat', 'qap_n1000_geo_id8.dat'
]

log_dir = Path('log')

configs = {
    'GD 231101': 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231101.log',
    'GD 231416': 'results_0.01_qap_gd+k+5_synthetic_0_20251021_231416.log',
    'GD 151557': 'results_0.01_qap_gd+k+5_synthetic_0_20251022_151557.log',
    'PGD 091202': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091202.log',
    'PGD 091943': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_091943.log',
    'PGD 152104 (FAQ)': 'results_0.01_qap_pgd+k+5_synthetic_0_20251022_152104.log',
}

for name, filename in configs.items():
    log_path = log_dir / filename
    with open(log_path, 'r') as f:
        content = f.read()

    found = []
    missing = []
    for inst in completed_instances:
        if inst in content:
            found.append(inst.replace('qap_n1000_geo_', '').replace('.dat', ''))
        else:
            missing.append(inst.replace('qap_n1000_geo_', '').replace('.dat', ''))

    print(f"\n{name}:")
    print(f"  Found ({len(found)}): {', '.join(found)}")
    if missing:
        print(f"  Missing ({len(missing)}): {', '.join(missing)}")
