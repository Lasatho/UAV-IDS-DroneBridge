#!/usr/bin/env python3
"""Remaining time of a manifest from the model flight time (split_manifest.estimate_s).

usage: eta_model.py <repo> <manifest> "<since local YYYY-MM-DD HH:MM>"
Wall/model ratio = wall time since t0 / model time of the missions started and
finished since t0 (t0 = since, or the first such mission start if it is more than
1 h later, e.g. a run started inside the window). Includes everything between
missions (SITL reboot, ground phase, restarts after degradation). Unlike the
median of attack_progress.py it does not depend on the order in which short and
long missions are flown.
"""
import csv, json, sys, time
from pathlib import Path

repo = Path(sys.argv[1]); ds = repo / "dataset"
sys.path.insert(0, str(repo / "data"))
from split_manifest import estimate_s  # noqa: E402

since = time.mktime(time.strptime(sys.argv[3], "%Y-%m-%d %H:%M"))
rows = list(csv.DictReader(open(ds / Path(sys.argv[2]).name)))


def est(r):
    m = json.loads((ds / "meta" / f"{r['mission_id']}_params.json").read_text())
    return estimate_s(ds / "waypoints" / r["waypoint_file"], m["drone_params"])


done = []
for r in rows:
    p = ds / "phase_labels" / f"{r['mission_id']}.json"
    if r["status"] == "completed" and p.exists():
        st = json.loads(p.read_text())["start_time"]
        if st >= since:
            done.append((st, r))
pend = [r for r in rows if r["status"] == "pending"]
now = time.time()
mrest = sum(est(r) for r in pend)
if not done:
    print(f"{Path(sys.argv[2]).name}: no mission since {sys.argv[3]}, {len(pend)} pending (model {mrest/86400:.2f} d)")
    sys.exit(0)
t0 = min(st for st, _ in done)
t0 = since if t0 - since < 3600 else t0
mdone = sum(est(r) for _, r in done)
ratio = (now - t0) / mdone
print(f"{Path(sys.argv[2]).name}: {len(done)} done since {time.strftime('%m-%d %H:%M', time.localtime(t0))} "
      f"in {(now-t0)/3600:.1f} h (model {mdone/3600:.1f} h, wall/model {ratio:.2f}), {len(pend)} pending "
      f"(model {mrest/86400:.2f} d) -> {mrest*ratio/86400:.1f} d, "
      f"end ~{time.strftime('%Y-%m-%d %H:%M', time.localtime(now + mrest*ratio))}")
