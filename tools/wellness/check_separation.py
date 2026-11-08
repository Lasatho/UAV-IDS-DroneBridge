#!/usr/bin/env python3
"""Separation check for parallel simulation instances (read-only).

For every completed mission with a phase label (instance, radio_freq_mhz; old labels
without these fields are instance 0 on 2437 MHz):
  1. pcap_onboard/ (radio of the instance): every received frame on the own channel,
     no frame of another channel; DroneBridge frames with continuous seq_num.
  2. pcap/ (hwsim0, all channels): frames on the own channel present; after the channel
     filter one sender per direction (seq_num continuous, loss below LOSS_MAX).
  3. Telemetry: GCS view and on-board view describe the same vehicle (median distance
     of GLOBAL_POSITION_INT at matching times below POS_MAX_M), on-board stream without
     position jumps (> JUMP_MAX_M between consecutive samples), start near home.
  4. Missions of different instances overlapping in time: the on-board capture of each
     holds none of the other's frames (covered by 1).
Prints one line per mission: OK or FLAG with reasons; exit 1 if any FLAG.
usage: check_separation.py <dataset> [--since-min N | mission_id ...] [--state FILE]
"""
import argparse
import csv
import json
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import pcap_to_csv as P  # noqa: E402

LOSS_MAX = 0.02      # share of lost DroneBridge frames per direction after the filter
POS_MAX_M = 3.0      # median distance GCS view vs on-board view
JUMP_MAX_M = 25.0    # position step between consecutive on-board samples (4 Hz)
HOME_MAX_M = 15.0    # first on-board position to home


def db_frames(path: Path, freq=None):
    """(freq, direction, comm_id, seq) of the valid DroneBridge frames, plus all freqs."""
    out, freqs = [], {}
    for _, b, orig_len in P.parse_pcap_frames(path):
        f = P.radiotap_freq_mhz(b)
        freqs[f] = freqs.get(f, 0) + 1
        if freq is not None and f != freq:
            continue
        fld = P.extract_db_fields(b, orig_len)
        if fld:
            out.append((f, fld["direction"], fld["comm_id"], fld["seq_num"]))
    return out, freqs


def loss(frames):
    """Share of lost frames per direction from seq_num gaps (mod 256)."""
    res = {}
    for d in sorted({x[1] for x in frames}):
        s = [x[3] for x in frames if x[1] == d]
        if len(s) < 2:
            continue
        lost = sum(((b - a) % 256) - 1 for a, b in zip(s, s[1:]) if (b - a) % 256 != 0)
        res[d] = lost / (len(s) + lost)
    return res


def positions(path: Path):
    rows = list(csv.DictReader(open(path)))
    return [(int(r["host_timestamp_ns"]), float(r["lat"]) / 1e7, float(r["lon"]) / 1e7) for r in rows]


def dist_m(a, b):
    k = 111320.0
    return math.hypot((a[1] - b[1]) * k, (a[2] - b[2]) * k * math.cos(math.radians(a[1])))


def check(ds: Path, m: str):
    lab = json.loads((ds / "phase_labels" / f"{m}.json").read_text())
    inst = lab.get("instance") or "0"
    freq = int(lab.get("radio_freq_mhz") or 2437)
    why, info = [], []
    # 1. on-board capture
    po = ds / "pcap_onboard" / f"{m}.pcap"
    if po.exists():
        fr, freqs = db_frames(po)
        foreign = {f: n for f, n in freqs.items() if f is not None and f != freq}
        if foreign:
            why.append(f"onboard foreign channel {foreign}")
        rx = [x for x in fr if x[0] == freq]
        lo = loss(rx)
        bad = {d: round(v, 3) for d, v in lo.items() if v > LOSS_MAX}
        if bad and lab.get("attack_type") in (None, "none", ""):
            why.append(f"onboard seq loss {bad}")
        info.append(f"onb {len(fr)} fr")
    else:
        why.append("no pcap_onboard")
    # 2. medium capture, filtered by channel
    pg = ds / "pcap" / f"{m}.pcap"
    if pg.exists():
        fr, freqs = db_frames(pg, freq)
        if not fr:
            why.append(f"pcap: no frames on {freq} MHz {freqs}")
        lo = loss(fr)
        bad = {d: round(v, 3) for d, v in lo.items() if v > LOSS_MAX}
        if bad and lab.get("attack_type") in (None, "none", ""):
            why.append(f"pcap seq loss after filter {bad} (second sender?)")
        others = {f: n for f, n in freqs.items() if f != freq}
        info.append(f"pcap own {len(fr)} other ch {others or 0}")
    else:
        why.append("no pcap")
    # 3. telemetry of both views
    tg = ds / "telemetry" / f"{m}_GLOBAL_POSITION_INT.csv"
    to = ds / "telemetry_onboard" / f"{m}_GLOBAL_POSITION_INT.csv"
    if tg.exists() and to.exists():
        g, o = positions(tg), positions(to)
        if o:
            jumps = max((dist_m(a, b) for a, b in zip(o, o[1:])), default=0)
            if jumps > JUMP_MAX_M:
                why.append(f"onboard position jump {jumps:.0f} m")
        if g:
            # two vehicles in the GCS stream (frames of another instance forwarded by db_gcs)
            gj = max((dist_m(a, b) for a, b in zip(g, g[1:]) if b[0] - a[0] < 2_000_000_000), default=0)
            if gj > JUMP_MAX_M:
                why.append(f"GCS-view position jump {gj:.0f} m")
            # home: first sample of the GCS view (same SITL home for all missions)
            if g and dist_m(o[0], g[0]) > HOME_MAX_M:
                why.append(f"onboard start {dist_m(o[0], g[0]):.0f} m from GCS start")
            gt = [x[0] for x in g]
            import bisect
            d = []
            for x in o[::8]:
                i = bisect.bisect_left(gt, x[0])
                c = [g[j] for j in (i - 1, i) if 0 <= j < len(g) and abs(g[j][0] - x[0]) < 300_000_000]
                if c:
                    d.append(min(dist_m(x, y) for y in c))
            if d:
                med = sorted(d)[len(d) // 2]
                info.append(f"pos med {med:.1f} m")
                if med > POS_MAX_M:
                    why.append(f"GCS vs onboard position median {med:.1f} m")
    else:
        why.append("telemetry missing")
    return inst, freq, why, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("missions", nargs="*")
    ap.add_argument("--since-min", type=float, default=None)
    ap.add_argument("--state", type=Path, default=None, help="file of already checked missions")
    a = ap.parse_args()
    done = set(a.state.read_text().split()) if a.state and a.state.exists() else set()
    ms = a.missions
    if a.since_min is not None:
        t0 = time.time() - a.since_min * 60
        ms = sorted(p.stem for p in (a.dataset / "phase_labels").glob("mission_*.json")
                    if p.stat().st_mtime > t0)
    ms = [m for m in ms if m not in done]
    flagged = 0
    for m in ms:
        try:
            inst, freq, why, info = check(a.dataset, m)
        except Exception as e:  # report, never stop the monitor
            inst, freq, why, info = "?", 0, [f"check error {type(e).__name__}: {e}"], []
        flagged += bool(why)
        print(f"{m} inst {inst} {freq} MHz {'FLAG ' + '; '.join(why) if why else 'OK'} | {', '.join(info)}",
              flush=True)
        if a.state and not why:
            with open(a.state, "a") as f:
                f.write(m + "\n")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
