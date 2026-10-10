#!/usr/bin/env python3
"""
sanity_check_v2.py — per-mission sanity check of simulated v2 missions
(normal dataset) on a simulation host. Standard library only, so it runs
with the host's system python next to the running simulation.

Checks per completed mission (or the missions given with --ids):
  - label/meta present, meta matches manifest, wind label (seed, turbulence,
    direction) matches meta, mission_success, timed_out
  - all 5 telemetry files: monotonic, no gap > 1 s, span ~ duration
  - ground phase before takeoff: sim time / wall time (RTF drop on the ground
    distorts timing and pcap), samples without GPS fix
  - start position < 5 m from home
  - presets: horizontal speed p95, climb max, max altitude
  - wind series within [min, max], gaps, mean direction
  - landing: tilt after touchdown from RAW_IMU (tipped-over vehicle),
    crash disarm in the label
  - RTF samples of the watchdog sampler log during the mission
  - pcap packet count

Usage (from the repo root on the host):
    python3 data/sanity_check_v2.py --manifest dataset/missions_manifest_v2_hp.csv \
        --rtf-log ~/sim_degradation.log [--ids mission_30104 ...] [--json out.json]
Exit code 1 if any mission has an issue.
"""

import argparse
import csv
import json
import math
import statistics
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

STREAMS = ["GLOBAL_POSITION_INT", "GPS_RAW_INT", "RAW_IMU",
           "SCALED_PRESSURE", "SERVO_OUTPUT_RAW"]
PRESET_SPEED = {"conservative": 3, "standard": 5, "dynamic": 10, "fast": 12}
# Sampler log is written in local time (CEST)
RTF_LOG_TZ = timezone(timedelta(hours=2))

MAX_GAP_S = 1.0
MIN_GROUND_RTF = 0.75     # sim/wall of the ground phase; normal ~0.8-1.0
MAX_START_DIST_M = 5.0
MAX_TILT_DEG = 15.0
MIN_RTF = 0.75            # watchdog threshold


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_rtf(path):
    samples = []
    if not path or not Path(path).exists():
        return samples
    for line in open(path, errors="replace"):
        if "real_time_factor:" not in line:
            continue
        try:
            t = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            v = float(line.rsplit("real_time_factor:", 1)[1].split()[0])
        except ValueError:
            continue
        samples.append((t.replace(tzinfo=RTF_LOG_TZ).timestamp(), v))
    return samples


def pcap_packets(path):
    n = 0
    with open(path, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            return 0
        endian = "<" if hdr[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        while True:
            rec = f.read(16)
            if len(rec) < 16:
                return n
            f.seek(struct.unpack(endian + "IIII", rec)[2], 1)
            n += 1


def distance_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371000 * math.asin(math.sqrt(a))


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def host_t(row):
    return int(row["host_timestamp_ns"]) / 1e9


def check_mission(ds, m, rtf):
    mid = m["mission_id"]
    issues = []
    r = {"id": mid, "preset": m["drone_preset"], "wind": m["wind_profile"]}
    try:
        lab = json.load(open(ds / "phase_labels" / f"{mid}.json"))
        meta = json.load(open(ds / "meta" / f"{mid}_params.json"))
    except (OSError, ValueError) as e:
        return {**r, "issues": [f"label/meta missing: {e}"]}

    r["duration_s"] = round(lab["duration_s"], 1)
    r["forced_disarm"] = bool(lab.get("landed_by_orchestrator"))
    if lab.get("timed_out"):
        issues.append("timed_out")
    if not lab["mission_success"]:
        issues.append("mission_success false")
    if str(lab.get("disarm_reason") or "").startswith("Crash"):
        issues.append(f"crash: {lab['disarm_reason']}")
    if (meta["drone_preset"] != m["drone_preset"]
            or int(meta["wind_direction_deg"]) != int(float(m["wind_direction_deg"]))):
        issues.append("meta != manifest")
    w = lab.get("wind") or {}
    seeds = [int(mid.split("_")[1]) + 100000 * k for k in range(3)]
    if (w.get("seed") not in seeds
            or w.get("turbulence") != meta["wind_params"]["turbulence"]
            or int(w.get("direction_deg", -1)) != int(meta["wind_direction_deg"])):
        issues.append("wind label != meta")
    if w.get("crashed_seeds"):
        r["crashed_seeds"] = w["crashed_seeds"]

    rs = [v for t, v in rtf if lab["start_time"] - 60 <= t <= lab["end_time"] + 60]
    r["rtf_min"] = min(rs) if rs else None
    # Single low samples coincide with the SITL reboot between missions
    # (lockstep), only repeated ones indicate a degradation
    low = [v for v in rs if v < MIN_RTF]
    if len(low) >= 2:
        issues.append(f"RTF < {MIN_RTF} in {len(low)} samples (min {min(low):.2f})")

    tel = {}
    for s in STREAMS:
        p = ds / "telemetry" / f"{mid}_{s}.csv"
        if not p.exists():
            issues.append(f"{s} missing")
            continue
        rows = read_csv(p)
        tel[s] = rows
        ts = [host_t(x) for x in rows]
        if len(ts) < 10:
            issues.append(f"{s}: only {len(ts)} rows")
            continue
        dt = [b - a for a, b in zip(ts, ts[1:])]
        if min(dt) <= 0:
            issues.append(f"{s}: non-monotonic")
        if max(dt) > MAX_GAP_S:
            issues.append(f"{s}: gap {max(dt):.1f} s")
        if abs((ts[-1] - ts[0]) - lab["duration_s"]) > 30:
            issues.append(f"{s}: span {ts[-1] - ts[0]:.0f} s vs {lab['duration_s']:.0f} s")
        r[f"hz_{s}"] = round((len(ts) - 1) / (ts[-1] - ts[0]), 2)

    g = tel.get("GLOBAL_POSITION_INT")
    if g and len(g) >= 10:
        home = meta["home"]
        r["start_dist_m"] = round(distance_m(int(g[0]["lat"]) / 1e7, int(g[0]["lon"]) / 1e7,
                                             home["lat"], home["lng"]), 2)
        if r["start_dist_m"] > MAX_START_DIST_M:
            issues.append(f"start {r['start_dist_m']:.0f} m from home")
        air = [i for i, x in enumerate(g) if int(x["relative_alt"]) > 1000]
        if not air:
            issues.append("never airborne")
        else:
            pre = g[:air[0] + 1]
            wall = host_t(pre[-1]) - host_t(pre[0])
            sim = (int(pre[-1]["time_boot_ms"]) - int(pre[0]["time_boot_ms"])) / 1000
            r["ground_wall_s"] = round(wall, 1)
            r["ground_rtf"] = round(sim / wall, 2) if wall > 0 else None
            if wall > 0 and sim / wall < MIN_GROUND_RTF:
                issues.append(f"ground phase sim/wall {sim / wall:.2f}")
            above = [x for x in g if int(x["relative_alt"]) > 5000]
            hs = [math.hypot(int(x["vx"]), int(x["vy"])) / 100 for x in above]
            r["hspeed_p95"] = round(quantile(hs, 0.95), 2)
            r["climb_max"] = round(-min(int(x["vz"]) for x in g) / 100, 2)
            r["alt_max"] = round(max(int(x["relative_alt"]) for x in g) / 1000, 1)
            limit = PRESET_SPEED[m["drone_preset"]] * 1.2 + 2 + float(m["wind_speed_max"])
            if hs and r["hspeed_p95"] > limit:
                issues.append(f"hspeed p95 {r['hspeed_p95']} implausible")
            # Landing: tilt from the gravity vector 2 s after touchdown
            last = air[-1]
            td = next((i for i in range(last, len(g)) if int(g[i]["relative_alt"]) < 200), None)
            imu = tel.get("RAW_IMU")
            if td is not None and imu:
                after = [x for x in imu if host_t(x) > host_t(g[td]) + 2]
                if after:
                    ax = statistics.mean(int(x["xacc"]) for x in after)
                    ay = statistics.mean(int(x["yacc"]) for x in after)
                    az = statistics.mean(int(x["zacc"]) for x in after)
                    r["tilt_after_landing_deg"] = round(math.degrees(
                        math.atan2(math.hypot(ax, ay), -az)), 1)
                    if r["tilt_after_landing_deg"] > MAX_TILT_DEG:
                        issues.append(f"tilted {r['tilt_after_landing_deg']:.0f} deg after landing")

    gps = tel.get("GPS_RAW_INT")
    if gps:
        nofix = sum(1 for x in gps if int(x["fix_type"]) < 3)
        if nofix:
            issues.append(f"{nofix} GPS samples without fix")

    wp = ds / "wind" / f"{mid}_wind.csv"
    if wp.exists():
        wr = read_csv(wp)
        sp = [float(x["speed"]) for x in wr]
        if sp:
            r["wind_mean"] = round(statistics.mean(sp), 2)
            r["wind_sd"] = round(statistics.pstdev(sp), 2)
            lo, hi = float(m["wind_speed_min"]), float(m["wind_speed_max"])
            if min(sp) < lo - 1e-3 or max(sp) > hi + 1e-3:
                issues.append("wind outside range")
            wt = [float(x["host_time"]) for x in wr]
            if len(wt) > 1 and max(b - a for a, b in zip(wt, wt[1:])) > 0.6:
                issues.append("wind series gap")
            dev = statistics.mean(((float(x["direction_deg"]) - meta["wind_direction_deg"] + 180)
                                   % 360) - 180 for x in wr)
            if abs(dev) > 25:
                issues.append(f"wind direction off by {dev:.0f} deg")
    else:
        issues.append("wind series missing")

    pp = ds / "pcap" / f"{mid}.pcap"
    if pp.exists():
        r["pcap_packets"] = pcap_packets(pp)
        if r["pcap_packets"] < 100:
            issues.append(f"pcap only {r['pcap_packets']} packets")
    else:
        issues.append("pcap missing")

    r["issues"] = issues
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="default: directory of the manifest")
    ap.add_argument("--rtf-log", default=str(Path.home() / "sim_degradation.log"))
    ap.add_argument("--ids", nargs="*", help="check only these missions (any status)")
    ap.add_argument("--json", type=Path, help="write per-mission results here")
    args = ap.parse_args()

    ds = args.dataset_dir or args.manifest.parent
    manifest = read_csv(args.manifest)
    if args.ids:
        missions = [m for m in manifest if m["mission_id"] in set(args.ids)]
    else:
        missions = [m for m in manifest if m["status"] == "completed"]
    rtf = read_rtf(args.rtf_log)

    status = {}
    for m in manifest:
        status[m["status"]] = status.get(m["status"], 0) + 1
    print(f"Manifest {args.manifest.name}: {status}")
    print(f"Checking {len(missions)} missions, {len(rtf)} RTF samples\n")

    results = [check_mission(ds, m, rtf) for m in missions]
    bad = [r for r in results if r["issues"]]
    for r in bad:
        print(f"{r['id']} {r['preset']} {r['wind']}: " + "; ".join(r["issues"]))

    print(f"\n{len(bad)} of {len(results)} missions with issues")
    for wind in ["calm", "light", "moderate", "gusty"]:
        rs = [r for r in results if r["wind"] == wind and "wind_mean" in r]
        if rs:
            forced = sum(r.get("forced_disarm", False) for r in rs)
            print(f"  {wind:8s} n={len(rs):4d}  wind mean {statistics.mean(r['wind_mean'] for r in rs):.2f}"
                  f"  sd {statistics.mean(r['wind_sd'] for r in rs):.2f}  forced disarm {forced}")
    for preset in PRESET_SPEED:
        rs = [r for r in results if r["preset"] == preset and "hspeed_p95" in r]
        if rs:
            print(f"  {preset:12s} n={len(rs):4d}  hspeed p95 median "
                  f"{statistics.median(r['hspeed_p95'] for r in rs):.1f}"
                  f"  climb max {max(r['climb_max'] for r in rs):.1f}"
                  f"  alt max {max(r['alt_max'] for r in rs):.0f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
