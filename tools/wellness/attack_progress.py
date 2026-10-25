#!/usr/bin/env python3
"""Read-only progress of an attack manifest with a remaining-time estimate.

usage: attack_progress.py <repo> <manifest>
Remaining time = for every pending mission the median recorded duration of
finished missions in the same attack x preset x wind cell (fallback attack x
preset, then attack, then all), plus the median gap between two missions.
"""
import csv, json, statistics, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

repo = Path(sys.argv[1]); ds = repo / "dataset"
rows = list(csv.DictReader(open(ds / sys.argv[2] if "/" not in sys.argv[2] else sys.argv[2])))
print("status", dict(Counter(r["status"] for r in rows)))
print("completed per attack", dict(Counter(r["attack_type"] for r in rows if r["status"] == "completed")),
      "of", dict(Counter(r["attack_type"] for r in rows)))
lab = {}
for r in rows:
    p = ds / "phase_labels" / f"{r['mission_id']}.json"
    if r["status"] in ("completed", "failed") and p.exists():
        lab[r["mission_id"]] = json.loads(p.read_text())
ends = sorted((L["start_time"], L["end_time"]) for L in lab.values())
gaps = [b[0] - a[1] for a, b in zip(ends, ends[1:]) if 0 < b[0] - a[1] < 600]
gap = statistics.median(gaps) if gaps else 84.0
cells = [defaultdict(list) for _ in range(4)]
keys = lambda r: [(r["attack_type"], r["drone_preset"], r["wind_profile"]),
                  (r["attack_type"], r["drone_preset"]), (r["attack_type"],), ()]
for r in rows:
    L = lab.get(r["mission_id"])
    if L and r["status"] == "completed":
        for c, k in zip(cells, keys(r)):
            c[k].append(L["duration_s"])
done_s = sum(L["duration_s"] for L in lab.values()) + gap * len(lab)
rem = 0.0
for r in rows:
    if r["status"] != "pending":
        continue
    for c, k in zip(cells, keys(r)):
        if c.get(k):
            rem += statistics.median(c[k]) + gap
            break
    else:
        rem += 600 + gap
n_done = sum(r["status"] == "completed" for r in rows)
print(f"done {n_done}/{len(rows)} = {100 * n_done / len(rows):.1f} %, "
      f"time done {done_s / 86400:.2f} d, remaining {rem / 86400:.2f} d "
      f"(= {100 * done_s / (done_s + rem):.1f} % by time), gap between missions {gap:.0f} s, "
      f"end ~{(datetime.now() + timedelta(seconds=rem)):%Y-%m-%d %H:%M}")
print("median duration s per attack", {a: round(statistics.median(v)) for (a,), v in
                                       ((k, v) for k, v in cells[2].items())})
