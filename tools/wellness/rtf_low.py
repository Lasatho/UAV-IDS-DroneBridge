#!/usr/bin/env python3
"""Read-only: RTF samples below threshold since a timestamp (sim_degradation.log)."""
import sys, re
since, thr = sys.argv[2], float(sys.argv[3])
for line in open(sys.argv[1]):
    if line[:16] < since: continue
    m = re.search(r"real_time_factor:([0-9.eE+-]+)", line)
    if m and float(m.group(1)) < thr: print(line[:19], round(float(m.group(1)), 2))
