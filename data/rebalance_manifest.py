#!/usr/bin/env python3
"""
rebalance_manifest.py — move pending missions from one host's manifest to
another's while a dataset run is in progress.

The time model of split_manifest.py does not include wind. Measured on the
v2 run, the ratio of actual to modelled duration depends strongly on the
wind profile (calm ~0.9, moderate ~2.3), so the initial split left the
windy missions on one host. This script re-estimates every pending mission
with a per-profile factor taken from the completed missions of the source
host and assigns them longest-first to the host that finishes earliest,
starting from the remaining work of each host.

Step 1 (on the source host, dry run without --apply):
    python3 data/rebalance_manifest.py \
        --manifest dataset/missions_manifest_v2_hp.csv \
        --dest-backlog-h 24 --keep mission_30512 \
        [--apply --rows-out /tmp/moved_rows.csv]
    --apply sets the status of the moved missions to MOVED_STATUS in the
    source manifest (backup *.bak_<date>_before_rebalance) and writes their
    rows with status "pending" to --rows-out.
Step 2 (on the destination host): append those rows to its manifest. The
    orchestrator only reads the manifest at start, so both orchestrators
    need a restart to pick up the change.

Missions given with --keep (e.g. the one in flight) stay on the source.
"""

import argparse
import csv
import json
import shutil
import time
from datetime import date, datetime
from pathlib import Path

from split_manifest import estimate_s

MOVED_STATUS = "reassigned_tr"
OVERHEAD_MEASURED_S = 84.0   # wall time between missions on HP, 48 h (2026-10-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--dest-backlog-h", type=float, required=True,
                    help="remaining work of the destination host in hours")
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rows-out", type=Path)
    args = ap.parse_args()
    ds = args.dataset_dir

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    by_id = {r["mission_id"]: r for r in rows}

    def model_s(r):
        meta = json.loads((ds / "meta" / f"{r['mission_id']}_params.json").read_text())
        return estimate_s(ds / "waypoints" / r["waypoint_file"], meta["drone_params"])

    # actual / modelled duration per wind profile from completed missions
    acc = {}
    for p in (ds / "phase_labels").glob("mission_3*.json"):
        lab = json.loads(p.read_text())
        r = by_id.get(lab["run_id"])
        if r is None or not lab.get("mission_success"):
            continue
        a = acc.setdefault(r["wind_profile"], [0.0, 0.0])
        a[0] += lab["duration_s"]
        a[1] += model_s(r)
    factor = {w: d / m for w, (d, m) in acc.items()}

    pending = [r for r in rows if r["status"] == "pending" and r["mission_id"] not in args.keep]
    est = {r["mission_id"]: model_s(r) * factor[r["wind_profile"]] + OVERHEAD_MEASURED_S
           for r in pending}
    load = {"src": sum(model_s(by_id[m]) * factor[by_id[m]["wind_profile"]] + OVERHEAD_MEASURED_S
                       for m in args.keep if by_id[m]["status"] == "pending"),
            "dst": args.dest_backlog_h * 3600}
    assign = {"src": [], "dst": []}
    for r in sorted(pending, key=lambda r: -est[r["mission_id"]]):
        h = min(load, key=load.get)
        assign[h].append(r)
        load[h] += est[r["mission_id"]]

    now = time.time()
    print("factor actual/model per wind:", {w: round(v, 2) for w, v in factor.items()})
    for h, rs in assign.items():
        mix = {}
        for r in rs:
            k = r["wind_profile"]
            mix[k] = mix.get(k, 0) + 1
        print(f"{h}: {len(rs)} missions {dict(sorted(mix.items()))}, "
              f"done ~{datetime.fromtimestamp(now + load[h]):%Y-%m-%d %H:%M} "
              f"({load[h] / 86400:.1f} d)")

    if not args.apply:
        return
    if args.rows_out is None:
        ap.error("--apply needs --rows-out")
    moved = {r["mission_id"] for r in assign["dst"]}
    bak = args.manifest.with_name(f"{args.manifest.name}.bak_{date.today():%Y%m%d}_before_rebalance")
    shutil.copy2(args.manifest, bak)
    with open(args.rows_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for r in assign["dst"]:
            w.writerow({**r, "status": "pending"})
    for r in rows:
        if r["mission_id"] in moved:
            r["status"] = MOVED_STATUS
    tmp = args.manifest.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(args.manifest)
    print(f"moved {len(moved)} -> {args.rows_out}, backup {bak}")


if __name__ == "__main__":
    main()
