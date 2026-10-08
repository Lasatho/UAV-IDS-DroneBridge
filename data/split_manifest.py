#!/usr/bin/env python3
"""
split_manifest.py — split a mission manifest across simulation hosts by
estimated simulation time (not by mission count: a conservative-preset
mission at 3 m/s takes ~2x as long as a fast one).

Time model per mission (wall clock):
    a * (horizontal path / WP_SPD + climb / WP_SPD_UP + descent / WP_SPD_DN
         + loiter time) + b + ground + overhead
including the RTL climb to RTL_ALT_M and the flight home. a, b, ground and
overhead were calibrated on the v1 normal dataset (flown with ArduCopter
defaults 10 / 2.5 / 1.5 m/s): a = 1.003, b = 44.7 s (median residual 11 s),
ground time in recording 40.5 s, per-mission overhead ~57 s (SITL reboot
with pose reset, GPS/EKF wait, parameter set, mission upload). Wind is not
modelled.

Missions are assigned longest-first to the host that is furthest below its
target share, so every host gets a mix of profiles and presets.

Usage:
    python3 split_manifest.py --manifest ../dataset/missions_manifest_v2.csv \
        --dataset-dir ../dataset --hosts hp=0.75 tr=0.25
    -> missions_manifest_v2_hp.csv, missions_manifest_v2_tr.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

HOME_LAT, HOME_LNG = -35.363261, 149.165230
K_LAT = 111320.0
K_LON = K_LAT * math.cos(math.radians(HOME_LAT))
A, B, GROUND_S, OVERHEAD_S = 1.003, 44.7, 40.5, 57.0


def estimate_s(waypoint_file: Path, params: dict) -> float:
    pts, loiter = [], 0.0
    for line in waypoint_file.read_text().split("\n")[1:]:
        p = line.split("\t")
        if len(p) < 12 or int(p[0]) == 0:
            continue
        cmd = int(p[3])
        if cmd == 22:                      # takeoff at home
            pts.append((0.0, 0.0, float(p[10])))
        elif cmd in (16, 19):              # waypoint, loiter_time
            pts.append(((float(p[8]) - HOME_LAT) * K_LAT,
                        (float(p[9]) - HOME_LNG) * K_LON, float(p[10])))
            if cmd == 19:
                loiter += float(p[4])
    h = up = dn = x = y = z = 0.0
    for n, e, alt in pts:
        h += math.hypot(n - x, e - y)
        up += max(0.0, alt - z)
        dn += max(0.0, z - alt)
        x, y, z = n, e, alt
    rtl_alt = params["RTL_ALT_M"]
    if z < rtl_alt:
        up += rtl_alt - z
        z = rtl_alt
    h += math.hypot(x, y)
    dn += z
    flight = (h / params["WP_SPD"] + up / params["WP_SPD_UP"]
              + dn / params["WP_SPD_DN"] + loiter)
    return A * flight + B + GROUND_S + OVERHEAD_S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hosts", nargs="+", required=True,
                        help="name=share, e.g. hp=0.75 tr=0.25")
    args = parser.parse_args()

    hosts = {h.split("=")[0]: float(h.split("=")[1]) for h in args.hosts}
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    est = {}
    for r in rows:
        meta = json.loads((args.dataset_dir / "meta" / f"{r['mission_id']}_params.json").read_text())
        est[r["mission_id"]] = estimate_s(
            args.dataset_dir / "waypoints" / r["waypoint_file"], meta["drone_params"])

    load = {h: 0.0 for h in hosts}
    assign = {h: [] for h in hosts}
    for r in sorted(rows, key=lambda r: -est[r["mission_id"]]):
        h = min(hosts, key=lambda h: load[h] / hosts[h])
        assign[h].append(r)
        load[h] += est[r["mission_id"]]

    stem = args.manifest.with_suffix("")
    for h, rs in assign.items():
        rs.sort(key=lambda r: r["mission_id"])
        out = Path(f"{stem}_{h}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rs)
        print(f"{h}: {len(rs)} missions, estimated {load[h] / 86400:.1f} days -> {out}")


if __name__ == "__main__":
    main()
