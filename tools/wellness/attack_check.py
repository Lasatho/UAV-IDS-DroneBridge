#!/usr/bin/env python3
"""Read-only sanity check of attack missions (attack dataset v2).

usage: attack_check.py <repo> <manifest> <clock_offset.json> <out.json> [since "YYYY-MM-DD HH:MM"]

Per finished mission (completed/failed):
  files       5 telemetry CSVs, pcap, wind series, label
  attack      fired, duration vs. schedule from attack_seed, start offset
              after AUTO, post-attack recording >= 30 s (unless timed out),
              c2 variant = meta, modes after the attack start, disarm reason
  telemetry   largest gap per stream, span vs. label duration; gaps on the
              ground before liftoff separately (pre_takeoff_gap_s)
  radio       frames/s per direction before / during / after the attack,
              foreign frames (no valid direction or comm_id != 200)
  gnss        GPS_RAW_INT fix/satellites and distance GPS_RAW_INT ->
              GLOBAL_POSITION_INT before / during the attack and in the last
              30 s; an issue if the GNSS is disturbed before the attack or
              still at the end (parameters of an earlier GNSS attack left
              set, Befund 33)
  landing     on its back on the ground while armed (RAW_IMU zacc > 0 for
              > 3 s below 0.5 m after having flown, before the disarm),
              force-disarm by the orchestrator's flip detection, and LAND or
              RTL after the attack that ends in the timeout (Befund 35)
Clocks: label and pcap are epoch, telemetry is time.monotonic_ns() of the
host; converted with the host's clock offset (epoch - monotonic).
"""
import csv, json, math, random, struct, sys, time
from datetime import datetime
from pathlib import Path

repo = Path(sys.argv[1]); ds = repo / "dataset"
manifest = repo / sys.argv[2] if not sys.argv[2].startswith("/") else Path(sys.argv[2])
off_ns = int(json.loads(Path(sys.argv[3]).read_text())["epoch_minus_monotonic_ns"])
out = Path(sys.argv[4])
since = time.mktime(datetime.strptime(sys.argv[5], "%Y-%m-%d %H:%M").timetuple()) if len(sys.argv) > 5 else 0
STREAMS = ["GLOBAL_POSITION_INT", "GPS_RAW_INT", "RAW_IMU", "SCALED_PRESSURE", "SERVO_OUTPUT_RAW"]
POST_S = 30.0
GNSS_DEV_M = 5.0  # GPS -> EKF distance of an undisturbed flight stays below
FLIPPED_HOLD_S = 3.0  # as the orchestrator's flip detection
PRE_TAKEOFF_GAP_S = 2.0  # telemetry gap on the ground before liftoff


def frames(path):
    with open(path, "rb") as f:
        g = f.read(24)
        if len(g) < 24:
            return
        e = "<" if g[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        res = 1e-9 if g[:4] in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d") else 1e-6
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            s, fr, incl, orig = struct.unpack(e + "IIII", h)
            d = f.read(incl)
            yield s + fr * res, d


def col(path, names):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = [[float(x[n]) for n in names] for x in r]
    return rows


def dist(lat1, lon1, lat2, lon2):
    k = 111320.0
    return math.hypot((lat1 - lat2) * k, (lon1 - lon2) * k * math.cos(math.radians(lat1)))


res = []
for r in csv.DictReader(open(manifest)):
    if r["status"] not in ("completed", "failed"):
        continue
    mid = r["mission_id"]; lab = ds / "phase_labels" / f"{mid}.json"
    if not lab.exists() or lab.stat().st_mtime < since:
        continue
    L = json.loads(lab.read_text())
    meta = json.loads((ds / "meta" / f"{mid}_params.json").read_text())
    o = {"id": mid, "status": r["status"], "attack": r["attack_type"],
         "variant": meta.get("attack_variant"), "preset": r["drone_preset"],
         "wind": r["wind_profile"], "dir": r["wind_direction_deg"], "issues": [],
         "finished": datetime.fromtimestamp(lab.stat().st_mtime).strftime("%m-%d %H:%M")}
    iss = o["issues"]
    if L.get("attack_type") != r["attack_type"]:
        iss.append(f"label attack_type {L.get('attack_type')}")
    if L.get("attack_variant") != meta.get("attack_variant"):
        iss.append(f"variant label {L.get('attack_variant')} != meta {meta.get('attack_variant')}")
    # files
    missing = [s for s in STREAMS if not (ds / "telemetry" / f"{mid}_{s}.csv").exists()]
    pcap = ds / "pcap" / f"{mid}.pcap"
    if missing: iss.append(f"missing telemetry {missing}")
    if not pcap.exists() or pcap.stat().st_size < 1000: iss.append("pcap missing/empty")
    if not (ds / "wind" / f"{mid}_wind.csv").exists(): iss.append("wind series missing")
    a0, a1 = L.get("attack_start"), L.get("attack_end")
    o["fired"] = a0 is not None and a1 is not None
    o["duration_s"] = round(L.get("duration_s", 0), 1)
    o["timed_out"] = L.get("timed_out")
    o["disarm_reason"] = L.get("disarm_reason")
    o["mission_success"] = L.get("mission_success")
    if not o["fired"]:
        iss.append("attack not fired")
        res.append(o); continue
    seed = meta["attack_seed"]; rng = random.Random(int(seed, 16) if isinstance(seed, str) else seed)
    rng.uniform(15, 45); sched = rng.uniform(15, 40)
    o["atk_dur"] = round(a1 - a0, 1); o["atk_sched"] = round(sched, 1)
    if not (sched - 0.5 <= a1 - a0 <= sched + 3.0):
        iss.append(f"attack duration {a1 - a0:.1f} s vs schedule {sched:.1f} s")
    auto = next((t for t, m in L.get("mode_changes", []) if m == "AUTO"), None)
    o["start_after_auto"] = round(a0 - auto, 1) if auto else None
    o["post_s"] = round(L["end_time"] - a1, 1)
    if o["post_s"] < POST_S - 1 and not L.get("timed_out"):
        iss.append(f"post-attack recording {o['post_s']} s < {POST_S}")
    o["modes_after"] = [m for t, m in L.get("mode_changes", []) if t >= a0]
    # telemetry
    try:
        span = None; gaps = {}
        for s in STREAMS:
            t = [x[0] / 1e9 for x in col(ds / "telemetry" / f"{mid}_{s}.csv", ["host_timestamp_ns"])]
            gaps[s] = round(max(b - a for a, b in zip(t, t[1:])), 2) if len(t) > 1 else None
            if s == "RAW_IMU" and t:
                span = (t[0] + off_ns / 1e9, t[-1] + off_ns / 1e9)
        o["max_gap_s"] = max(v for v in gaps.values() if v is not None)
        # rf_jamming cuts the radio link the telemetry is recorded over, and
        # the injected frames of frame_injection crowd it out: a gap that
        # overlaps the attack (for rf_jamming until the link is back) is the
        # attack effect, reported as link_loss_s, not as an issue
        gp = col(ds / "telemetry" / f"{mid}_GLOBAL_POSITION_INT.csv", ["host_timestamp_ns", "relative_alt"])
        t = [x[0] / 1e9 + off_ns / 1e9 for x in gp]
        big = [(a, b) for a, b in zip(t, t[1:]) if b - a > 1.0]
        in_atk = [(a, b) for a, b in big if a <= a1 + 1 and b >= a0 - 1]
        if r["attack_type"] in ("rf_jamming", "frame_injection"):
            o["link_loss_s"] = round(sum(b - a for a, b in in_atk), 1)
            big = [g for g in big if g not in in_atk]
        # a gap on the ground before liftoff (> 1 m) does not touch the flight
        # (HP, from mission ~40870 on, up to 1.14 s just after the logger
        # start): reported as pre_takeoff_gap_s, an issue only above 2 s
        lift = next((t[i] for i, x in enumerate(gp) if x[1] > 1000), None)
        pre = [g for g in big if lift is not None and g[1] <= lift and g not in in_atk]
        o["pre_takeoff_gap_s"] = round(max((b - a for a, b in pre), default=0.0), 2)
        if o["pre_takeoff_gap_s"] > PRE_TAKEOFF_GAP_S:
            iss.append(f"telemetry gap {o['pre_takeoff_gap_s']:.1f} s before takeoff")
        big = [g for g in big if g not in pre]
        for where, gs in (("in", [g for g in big if g in in_atk]),
                          ("outside", [g for g in big if g not in in_atk])):
            if gs:
                iss.append(f"telemetry gap {max(b - a for a, b in gs):.1f} s {where} the attack")
        if span and not (span[0] - 5 <= a0 and a1 <= span[1] + 1):
            iss.append("attack outside telemetry span (clock offset?)")
        # gnss
        g = col(ds / "telemetry" / f"{mid}_GPS_RAW_INT.csv", ["host_timestamp_ns", "fix_type", "satellites_visible", "lat", "lon"])
        p = col(ds / "telemetry" / f"{mid}_GLOBAL_POSITION_INT.csv", ["host_timestamp_ns", "lat", "lon", "relative_alt"])
        def window(rows, t0, t1):
            return [x for x in rows if t0 <= x[0] / 1e9 + off_ns / 1e9 <= t1]
        pi = 0; dev = {"pre": 0.0, "atk": 0.0, "end": 0.0}; fixmin = {}; satmin = {}
        for name, (t0, t1) in {"pre": (a0 - 30, a0), "atk": (a0, a1),
                               "end": (L["end_time"] - 30, L["end_time"])}.items():
            gw = window(g, t0, t1)
            fixmin[name] = min((x[1] for x in gw), default=None)
            satmin[name] = min((x[2] for x in gw), default=None)
            for x in gw:
                while pi + 1 < len(p) and p[pi + 1][0] <= x[0]:
                    pi += 1
                if x[1] >= 3 and pi < len(p):
                    dev[name] = max(dev[name], dist(x[3] / 1e7, x[4] / 1e7, p[pi][1] / 1e7, p[pi][2] / 1e7))
            pi = 0
        o["gps"] = {"fix_min": fixmin, "sat_min": satmin, "dev_max_m": {k: round(v, 1) for k, v in dev.items()}}
        # the end window needs time after the attack: after a spoofing reset
        # the GPS needs up to ~20 s to rejoin the EKF position
        for name in ("pre", "end") if o["post_s"] >= 60 else ("pre",):
            if dev[name] >= GNSS_DEV_M or (fixmin[name] is not None and fixmin[name] < 3):
                iss.append(f"GNSS disturbed {'before the attack' if name == 'pre' else 'at mission end'}: "
                           f"GPS->EKF {dev[name]:.1f} m, fix min {fixmin[name]}")
        # landing: replay of the orchestrator's flip detection up to the
        # disarm (or the end); armed on its back beyond its hold time means
        # the detection did not act (before its deployment: Befund 35)
        imu = col(ds / "telemetry" / f"{mid}_RAW_IMU.csv", ["host_timestamp_ns", "zacc"])
        stop = L.get("disarm_time") or L["end_time"]
        ev = sorted([(x[0], 0, x[3]) for x in p] + [(x[0], 1, x[1]) for x in imu])
        air = False; alt = None; back0 = None; on_back = 0.0
        for t, k, v in ev:
            t = t / 1e9 + off_ns / 1e9
            if t > stop:
                break
            if k == 0:
                alt = v / 1000.0; air = air or alt > 5.0
            elif air and alt is not None and alt < 0.5 and v > 0:
                back0 = back0 or t; on_back = max(on_back, t - back0)
            else:
                back0 = None
        o["on_back_armed_s"] = round(on_back, 1)
        if on_back > FLIPPED_HOLD_S + 1:
            iss.append(f"on its back on the ground while armed {on_back:.0f} s")
    except Exception as e:
        iss.append(f"telemetry read error {e}")
    if L.get("flipped_disarmed"):
        iss.append("flipped on touchdown, force-disarmed by the orchestrator")
    if L.get("timed_out") and set(o["modes_after"]) & {"LAND", "RTL"}:
        iss.append(f"timed out after {'/'.join(m for m in o['modes_after'] if m in ('LAND', 'RTL'))}")
    # radio
    try:
        cnt = {k: {"up": 0, "down": 0, "foreign": 0} for k in ("pre", "atk", "post")}
        for t, d in frames(pcap):
            ph = "pre" if t < a0 else ("atk" if t <= a1 else "post")
            if len(d) < 32 or d[26] not in (1, 3) or d[27] != 200:
                cnt[ph]["foreign"] += 1
            else:
                cnt[ph]["up" if d[26] == 1 else "down"] += 1
        dur = {"pre": max(a0 - L["start_time"], 1e-3), "atk": max(a1 - a0, 1e-3), "post": max(L["end_time"] - a1, 1e-3)}
        o["radio_per_s"] = {ph: {k: round(v / dur[ph], 2) for k, v in c.items()} for ph, c in cnt.items()}
    except Exception as e:
        iss.append(f"pcap read error {e}")
    res.append(o)

json.dump(res, open(out, "w"), indent=1)
print(f"{len(res)} missions checked, {sum(bool(o['issues']) for o in res)} with issues")
for o in res:
    if o["issues"]:
        print(o["id"], o["attack"], o["variant"] or "", "|", "; ".join(o["issues"]))
pre_gap = [o for o in res if o.get("pre_takeoff_gap_s", 0) > 1.0]
print(f"pre-takeoff telemetry gap > 1 s: {len(pre_gap)}, max {max((o['pre_takeoff_gap_s'] for o in pre_gap), default=0.0):.2f} s",
      " ".join(o["id"][8:] for o in pre_gap))
# per attack type summary
import statistics as st
by = {}
for o in res:
    by.setdefault((o["attack"], o["variant"]), []).append(o)
print("attack variant n fired | start_after_auto med | post_s med | modes_after (top) | radio up/down/foreign per s pre -> atk -> post (median) | gps fix_min atk min, dev_max atk med / pre med")
for k in sorted(by, key=str):
    os_ = [o for o in by[k] if o.get("fired")]
    if not os_:
        print(k, len(by[k]), "none fired"); continue
    med = lambda f: round(st.median(f(o) for o in os_), 2)
    modes = {}
    for o in os_:
        key = "->".join(o["modes_after"]) or "-"
        modes[key] = modes.get(key, 0) + 1
    rad = lambda ph, c: med(lambda o: o["radio_per_s"][ph][c]) if all("radio_per_s" in o for o in os_) else None
    gps = [o for o in os_ if "gps" in o]
    print(k[0], k[1] or "", len(by[k]), len(os_), "|", med(lambda o: o["start_after_auto"] or 0), "|", med(lambda o: o["post_s"]), "|",
          sorted(modes.items(), key=lambda x: -x[1])[:3], "|",
          "/".join(str(rad("pre", c)) for c in ("up", "down", "foreign")), "->",
          "/".join(str(rad("atk", c)) for c in ("up", "down", "foreign")), "->",
          "/".join(str(rad("post", c)) for c in ("up", "down", "foreign")), "|",
          min((o["gps"]["fix_min"]["atk"] for o in gps if o["gps"]["fix_min"]["atk"] is not None), default=None),
          round(st.median(o["gps"]["dev_max_m"]["atk"] for o in gps), 1) if gps else None, "/",
          round(st.median(o["gps"]["dev_max_m"]["pre"] for o in gps), 1) if gps else None)
