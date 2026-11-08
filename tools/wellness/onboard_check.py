#!/usr/bin/env python3
"""On-board view of re-flown attack missions (rf_jamming/frame_injection, 2026-11-03).

usage: onboard_check.py <repo> <manifest> <clock_offset.json> <out.json> [since "YYYY-MM-DD HH:MM"]
Completed missions whose label is newer than <since> and that have an
on-board recording. Per mission: on-board telemetry (5 CSVs) and radio capture
(pcap_onboard/) present, on-board cadence, largest on-board telemetry gap
overall and around the attack window (with the neighbouring samples), GCS gap
around the attack for comparison. Flags: missing files, on-board gap > 1 s,
on-board cadence outside 3.5-4.5 Hz, empty on-board capture.
"""
import csv, json, sys, time
from pathlib import Path

MSGS = ["GLOBAL_POSITION_INT", "GPS_RAW_INT", "RAW_IMU", "SCALED_PRESSURE", "SERVO_OUTPUT_RAW"]
repo = Path(sys.argv[1]); ds = repo / "dataset"
off = json.load(open(sys.argv[3]))["epoch_minus_monotonic_ns"]
since = time.mktime(time.strptime(sys.argv[5], "%Y-%m-%d %H:%M")) if len(sys.argv) > 5 else 0
rows = [r for r in csv.DictReader(open(ds / sys.argv[2] if "/" not in sys.argv[2] else sys.argv[2])) if r["status"] == "completed"]


def ts(p):
    with open(p) as f:
        return [int(x["host_timestamp_ns"]) for x in csv.DictReader(f)]


def gap_around(t, a0, a1):
    i0 = max(0, next((i for i, x in enumerate(t) if x >= a0), len(t)) - 1)
    i1 = next((i for i, x in enumerate(t) if x > a1), len(t) - 1)
    w = t[i0:i1 + 1]
    return max((b - a for a, b in zip(w, w[1:])), default=0) / 1e9


res, flags = [], []
for r in rows:
    m = r["mission_id"]; lp = ds / "phase_labels" / f"{m}.json"
    if not lp.exists() or lp.stat().st_mtime < since or not (ds / "telemetry_onboard" / f"{m}_RAW_IMU.csv").exists():
        continue
    lab = json.loads(lp.read_text())
    need = [ds / "telemetry_onboard" / f"{m}_{k}.csv" for k in MSGS] + [ds / "pcap_onboard" / f"{m}.pcap"]
    missing = [str(p.relative_to(ds)) for p in need if not p.exists()]
    d = {"mid": m, "type": r["attack_type"], "missing": missing}
    if not missing and lab.get("attack_start"):
        a0 = lab["attack_start"] * 1e9 - off; a1 = lab["attack_end"] * 1e9 - off
        on = {k: ts(ds / "telemetry_onboard" / f"{m}_{k}.csv") for k in MSGS}
        d["onboard_hz"] = round(min((len(t) - 1) / ((t[-1] - t[0]) / 1e9) for t in on.values()), 2)
        d["onboard_gap_s"] = round(max(max((b - a for a, b in zip(t, t[1:])), default=0) for t in on.values()) / 1e9, 2)
        d["onboard_gap_attack_s"] = round(max(gap_around(t, a0, a1) for t in on.values()), 2)
        d["gcs_gap_attack_s"] = round(gap_around(ts(ds / "telemetry" / f"{m}_RAW_IMU.csv"), a0, a1), 2)
        d["pcap_onboard_bytes"] = (ds / "pcap_onboard" / f"{m}.pcap").stat().st_size
    f = []
    if missing: f.append("missing " + ",".join(missing))
    elif "onboard_hz" in d:
        if d["onboard_gap_s"] > 1: f.append(f"on-board gap {d['onboard_gap_s']} s")
        if not 3.5 <= d["onboard_hz"] <= 4.5: f.append(f"on-board {d['onboard_hz']} Hz")
        if d["pcap_onboard_bytes"] < 1000: f.append("on-board capture empty")
    d["flags"] = f; res.append(d)
    if f: flags.append(d)
json.dump(res, open(sys.argv[4], "w"), indent=1)
print(f"checked {len(res)} re-flown missions, flagged {len(flags)}")
for t in sorted({d["type"] for d in res}):
    x = [d for d in res if d["type"] == t and "onboard_hz" in d]
    if x:
        print(f"  {t}: n {len(x)}, on-board gap around attack max {max(d['onboard_gap_attack_s'] for d in x)} s "
              f"(GCS max {max(d['gcs_gap_attack_s'] for d in x)} s), on-board cadence min {min(d['onboard_hz'] for d in x)} Hz")
for d in flags:
    print("!", d["mid"], d["type"], "; ".join(d["flags"]))
