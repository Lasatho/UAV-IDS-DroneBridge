"""Sim:wall in 10-s windows in flight (all samples > 3 m) and at touchdown.

usage: flight_rtf.py <repo> <manifest> <since 'YYYY-MM-DD HH:MM'>
Completed missions whose label is newer than <since>. Prints missions with
an in-flight window below 0.9, and a count summary.
"""
import csv, sys, time
from pathlib import Path

repo = Path(sys.argv[1]); ds = repo / "dataset"
since = time.mktime(time.strptime(sys.argv[3], "%Y-%m-%d %H:%M"))
rows = [r for r in csv.DictReader(open(repo / sys.argv[2])) if r["status"] == "completed"]
n = 0; flagged = []
for r in rows:
    m = r["mission_id"]
    lab = ds / "phase_labels" / f"{m}.json"
    if not lab.exists() or lab.stat().st_mtime < since:
        continue
    n += 1
    g = list(csv.DictReader(open(ds / "telemetry" / f"{m}_GLOBAL_POSITION_INT.csv")))
    wins, cur = [], []
    for x in g:
        if cur and int(x["host_timestamp_ns"]) - int(cur[0]["host_timestamp_ns"]) >= 10e9:
            wins.append(cur); cur = []
        cur.append(x)
    fl = []
    for w in wins:
        if len(w) < 2:
            continue
        dw = (int(w[-1]["host_timestamp_ns"]) - int(w[0]["host_timestamp_ns"])) / 1e9
        ds_ = (int(w[-1]["time_boot_ms"]) - int(w[0]["time_boot_ms"])) / 1e3
        if dw <= 0:
            continue
        alt = min(int(x["relative_alt"]) for x in w) / 1000
        if alt > 3:
            fl.append(ds_ / dw)
    mn = min(fl) if fl else None
    if mn is None or mn < 0.9:
        flagged.append((m, None if mn is None else round(mn, 2)))
print(f"checked {n}, in-flight window < 0.9: {len(flagged)}")
for f in flagged:
    print(*f)
