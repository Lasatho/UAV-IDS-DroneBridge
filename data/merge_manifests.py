#!/usr/bin/env python3
"""
merge_manifests.py — merge the per-host manifests of a split run into one.

Every mission of the master manifest must be `completed` on exactly one
host (rows handed to another host, e.g. `reassigned_tr`, are skipped). The
merged manifest keeps the master's columns and order, takes the status from
the host that flew the mission and adds a `host` column.

Usage:
    python3 merge_manifests.py --master ../dataset/missions_manifest_v2.csv \
        --hosts hp=hp.csv tr=tr.csv --out ../dataset/missions_manifest_v2_merged.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--hosts", nargs="+", required=True, help="name=manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.master, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        master = list(reader)

    flown = collections.defaultdict(list)
    for spec in args.hosts:
        host, path = spec.split("=", 1)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] == "completed":
                    flown[r["mission_id"]].append(host)
                elif not r["status"].startswith("reassigned"):
                    sys.exit(f"{host}: {r['mission_id']} has status {r['status']}")

    ids = {r["mission_id"] for r in master}
    extra = sorted(set(flown) - ids)
    missing = sorted(ids - set(flown))
    double = sorted(m for m, h in flown.items() if len(h) > 1)
    if extra or missing or double:
        sys.exit(f"not in master {extra[:5]}, not flown {missing[:5]}, "
                 f"flown twice {double[:5]}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["host"])
        w.writeheader()
        for r in master:
            w.writerow({**r, "status": "completed", "host": flown[r["mission_id"]][0]})
    print(f"{len(master)} missions -> {args.out}",
          dict(collections.Counter(h[0] for h in flown.values())))


if __name__ == "__main__":
    main()
