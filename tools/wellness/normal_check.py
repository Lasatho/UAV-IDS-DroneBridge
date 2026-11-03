#!/usr/bin/env python3
"""Sanity check of normal missions with GCS and on-board telemetry (dataset v3).

usage: normal_check.py <repo> <manifest> <out.json> [since "YYYY-MM-DD HH:MM"]
Completed missions whose label is newer than <since>. Per mission: files
(5 GCS CSVs, 5 on-board CSVs, pcap, wind series, label), success/timeout,
on-board cadence (rows/s on the host clock) and largest gap per stream for
both views, in flight (on-board GLOBAL_POSITION_INT relative_alt > 3 m) and
before takeoff. Flags: missing files, not successful, on-board gap > 1 s,
on-board cadence outside 3.5-4.5 Hz, GCS gap > 1 s in flight.
"""
import csv, json, statistics, sys, time
from pathlib import Path

MSGS = ["GLOBAL_POSITION_INT", "GPS_RAW_INT", "RAW_IMU", "SCALED_PRESSURE", "SERVO_OUTPUT_RAW"]
repo = Path(sys.argv[1]); ds = repo / "dataset"
since = time.mktime(time.strptime(sys.argv[4], "%Y-%m-%d %H:%M")) if len(sys.argv) > 4 else 0
rows = [r for r in csv.DictReader(open(ds / sys.argv[2])) if r["status"] == "completed"]


def ts(p):
    with open(p) as f:
        return [int(x["host_timestamp_ns"]) for x in csv.DictReader(f)]


def gaps(t, lo=None, hi=None):
    t = [x for x in t if (lo is None or x >= lo) and (hi is None or x <= hi)]
    return max((b - a for a, b in zip(t, t[1:])), default=0) / 1e9


res, flags = [], []
for r in rows:
    m = r["mission_id"]; lab_p = ds / "phase_labels" / f"{m}.json"
    if not lab_p.exists() or lab_p.stat().st_mtime < since:
        continue
    lab = json.loads(lab_p.read_text())
    need = [ds / "telemetry" / f"{m}_{k}.csv" for k in MSGS] + \
           [ds / "telemetry_onboard" / f"{m}_{k}.csv" for k in MSGS] + \
           [ds / "pcap" / f"{m}.pcap", ds / "wind" / f"{m}_wind.csv"]
    missing = [str(p.relative_to(ds)) for p in need if not p.exists()]
    d = {"mid": m, "preset": r["drone_preset"], "wind": r["wind_profile"], "success": lab["mission_success"],
         "timed_out": lab["timed_out"], "duration_s": round(lab["duration_s"]), "missing": missing}
    if not missing:
        g = list(csv.DictReader(open(ds / "telemetry_onboard" / f"{m}_GLOBAL_POSITION_INT.csv")))
        air = [int(x["host_timestamp_ns"]) for x in g if float(x["relative_alt"]) > 3000]
        t0, t1 = (air[0], air[-1]) if air else (None, None)
        on = {k: ts(ds / "telemetry_onboard" / f"{m}_{k}.csv") for k in MSGS}
        gc = {k: ts(ds / "telemetry" / f"{m}_{k}.csv") for k in MSGS}
        d["onboard_hz"] = round(min((len(t) - 1) / ((t[-1] - t[0]) / 1e9) for t in on.values()), 2)
        d["onboard_gap_s"] = round(max(gaps(t) for t in on.values()), 2)
        d["gcs_gap_air_s"] = round(max(gaps(t, t0, t1) for t in gc.values()), 2) if air else None
        d["gcs_gap_pre_s"] = round(max(gaps(t, None, t0) for t in gc.values()), 2) if air else None
        d["airborne_s"] = round((t1 - t0) / 1e9) if air else 0
    f = []
    if missing: f.append("missing " + ",".join(missing))
    if not d["success"]: f.append("not successful")
    if d["timed_out"]: f.append("timed out")
    if not missing:
        if d["onboard_gap_s"] > 1: f.append(f"on-board gap {d['onboard_gap_s']} s")
        if not 3.5 <= d["onboard_hz"] <= 4.5: f.append(f"on-board {d['onboard_hz']} Hz")
        if d["gcs_gap_air_s"] and d["gcs_gap_air_s"] > 1: f.append(f"GCS gap in flight {d['gcs_gap_air_s']} s")
        if not d["airborne_s"]: f.append("never airborne")
    d["flags"] = f; res.append(d)
    if f: flags.append(d)
json.dump(res, open(sys.argv[3], "w"), indent=1)
ok = [d for d in res if not d["missing"]]
print(f"checked {len(res)} missions since {sys.argv[4] if len(sys.argv) > 4 else 'start'}, flagged {len(flags)}")
if ok:
    med = lambda k: statistics.median(d[k] for d in ok if d.get(k) is not None)
    print(f"median: on-board {med('onboard_hz')} Hz, on-board max gap {med('onboard_gap_s')} s, "
          f"GCS max gap in flight {med('gcs_gap_air_s')} s, before takeoff {med('gcs_gap_pre_s')} s")
    print(f"max: on-board gap {max(d['onboard_gap_s'] for d in ok)} s, GCS gap in flight "
          f"{max(d['gcs_gap_air_s'] or 0 for d in ok)} s, before takeoff {max(d['gcs_gap_pre_s'] or 0 for d in ok)} s")
for d in flags:
    print("!", d["mid"], d["preset"], d["wind"], "; ".join(d["flags"]))
