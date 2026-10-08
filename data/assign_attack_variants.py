#!/usr/bin/env python3
"""
assign_attack_variants.py — write `attack_variant` into the meta JSON of
attack missions that have sub-variants (currently c2_hijacking).

c2_hijacking missions, sorted by mission_id, get land / rtl / terminate in
turn (672 missions -> 224 each). The mission order cycles through geometry,
drone preset and wind, so a period of 3 spreads the variants evenly across
them. Idempotent: rerunning writes the same values.

Usage:
    python3 assign_attack_variants.py --manifest ../dataset/missions_manifest_attacks.csv \
        --meta-dir ../dataset/meta
"""

import argparse
import collections
import csv
import json
from pathlib import Path

VARIANTS = {"c2_hijacking": ("land", "rtl", "terminate")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        missions = sorted(csv.DictReader(f), key=lambda r: r["mission_id"])

    counts = collections.Counter()
    for attack_type, variants in VARIANTS.items():
        ids = [m["mission_id"] for m in missions if m["attack_type"] == attack_type]
        for i, mission_id in enumerate(ids):
            path = args.meta_dir / f"{mission_id}_params.json"
            meta = json.loads(path.read_text())
            meta["attack_variant"] = variants[i % len(variants)]
            path.write_text(json.dumps(meta, indent=2))
            counts[(attack_type, meta["attack_variant"])] += 1
    for k, n in sorted(counts.items()):
        print(*k, n)


if __name__ == "__main__":
    main()
