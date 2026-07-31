#!/usr/bin/env python3
"""
flypaw_to_telemetry.py — adapts the reference FlyPaw/AERPAW dataset
(dataset/testing/flypaw/) into the CSV layout that
training/data/preprocessing.py expects, so the training pipeline
(discovery -> alignment -> windowing -> normalization -> training) can be
smoke-tested end to end on real flight recordings instead of simulated
SITL data.

FlyPaw has no IMU, barometer, or servo-channel data (see
dataset/testing/flypaw/README.md — it only records GPS/position,
battery and iperf link measurements). RAW_IMU, SCALED_PRESSURE and
SERVO_OUTPUT_RAW are therefore written as header-only placeholder files:
discover_missions() requires all five message-type files to be present
per mission, but load_mission() never opens a file for a message type
that is disabled in the config. training/configs/flypaw_test.yaml
disables exactly these three groups, so no sensor values are fabricated
or fed to a model.

Two source runs are converted:
  - 2022-01-13 (dense, ~1 Hz, ~18 min): *_vehicleOut.txt
    -> chunked into several missions, since split_missions() splits on
       mission level and needs more than one mission to produce a
       non-degenerate train/val/test split.
  - 2022-03-11 (sparse, ~1 sample / 8-10 s, ~5 min): telemetry_*.json
    -> single mission (too short to chunk further).

Position fields are re-encoded into the same integer MAVLink units
data/telemetry_logger.py writes (lat/lon in 1e7 deg, alt/relative_alt in
mm, vx/vy/vz in cm/s, hdg/cog in cdeg), so feature scale stays consistent
with real simulation-generated missions.

vx/vy/vz, vel and cog are derived from consecutive GPS fixes — the same
quantities a GPS receiver's own position/Doppler filter would output,
not engineered ML features. eph/epv (DOP-based uncertainty, not
derivable from position alone) are written as 65535, MAVLink's
"value unknown" sentinel for those fields.

Usage:
    python3 data/flypaw_to_telemetry.py
    python3 data/flypaw_to_telemetry.py --input-dir dataset/testing/flypaw \
        --output-dir dataset/testing/flypaw/telemetry --chunk-seconds 180
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

# Mirrors data/telemetry_logger.py LOGGED_MESSAGES / training/data/features.py
# MESSAGE_FEATURES — field name and order must match both exactly.
FIELDS: dict[str, list[str]] = {
    "GLOBAL_POSITION_INT": [
        "lat", "lon", "alt", "relative_alt", "vx", "vy", "vz", "hdg",
    ],
    "GPS_RAW_INT": [
        "fix_type", "lat", "lon", "alt", "eph", "epv", "vel", "cog",
        "satellites_visible",
    ],
    "RAW_IMU": [
        "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro",
        "xmag", "ymag", "zmag",
    ],
    "SCALED_PRESSURE": ["press_abs", "press_diff", "temperature"],
    "SERVO_OUTPUT_RAW": [f"servo{i}_raw" for i in range(1, 9)],
}
UNAVAILABLE_MESSAGES = ["RAW_IMU", "SCALED_PRESSURE", "SERVO_OUTPUT_RAW"]

UNKNOWN_U16 = 65535  # MAVLink "value unknown" sentinel (eph/epv/cog/vel)
DEG_TO_M = 111_320.0  # meters per degree latitude, equirectangular approx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_iso(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def derive_motion(fixes: list[tuple[float, float, float, float]]) -> list[dict]:
    """fixes: (t_epoch_s, lat_deg, lon_deg, alt_m) per point, time-ordered.

    Returns per-point vx/vy/vz (NED, cm/s), vel (ground speed, cm/s) and
    course-over-ground (cdeg), derived from consecutive GPS fixes. First
    point has no predecessor and gets zero motion.
    """
    out = []
    prev = fixes[0]
    for t, lat, lon, alt in fixes:
        pt, plat, plon, palt = prev
        dt = t - pt
        if dt <= 0:
            dx_n = dy_e = dz = 0.0
        else:
            dx_n = (lat - plat) * DEG_TO_M / dt
            dy_e = (lon - plon) * DEG_TO_M * math.cos(math.radians(lat)) / dt
            dz = -(alt - palt) / dt
        course = math.degrees(math.atan2(dy_e, dx_n)) % 360.0
        out.append(
            {
                "vx": round(dx_n * 100),
                "vy": round(dy_e * 100),
                "vz": round(dz * 100),
                "vel": round(math.hypot(dx_n, dy_e) * 100),
                "cog": round(course * 100),
            }
        )
        prev = (t, lat, lon, alt)
    return out


def write_csv(path: Path, msg_type: str, rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host_timestamp_ns"] + FIELDS[msg_type])
        w.writerows(rows)


def write_placeholders(out_dir: Path, mission_id: str) -> None:
    for msg_type in UNAVAILABLE_MESSAGES:
        path = out_dir / f"{mission_id}_{msg_type}.csv"
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(["host_timestamp_ns"] + FIELDS[msg_type])


# ---------------------------------------------------------------------------
# Mission writers
# ---------------------------------------------------------------------------

def write_mission(
    out_dir: Path,
    mission_id: str,
    points: list[dict],
) -> None:
    """points: dicts with t_epoch_s, lat, lon, alt_m, relative_alt_m,
    fix_type, satellites_visible."""
    fixes = [(p["t_epoch_s"], p["lat"], p["lon"], p["alt_m"]) for p in points]
    motion = derive_motion(fixes)

    gpi_rows, gps_rows = [], []
    for p, mo in zip(points, motion):
        host_ts_ns = round(p["t_epoch_s"] * 1e9)
        lat_e7 = round(p["lat"] * 1e7)
        lon_e7 = round(p["lon"] * 1e7)
        alt_mm = round(p["alt_m"] * 1000)
        rel_alt_mm = round(p["relative_alt_m"] * 1000)
        hdg_cdeg = p.get("hdg_cdeg")
        if hdg_cdeg is None:
            hdg_cdeg = mo["cog"]  # no compass reading -> GPS ground track

        gpi_rows.append(
            [host_ts_ns, lat_e7, lon_e7, alt_mm, rel_alt_mm,
             mo["vx"], mo["vy"], mo["vz"], hdg_cdeg]
        )
        gps_rows.append(
            [host_ts_ns, p["fix_type"], lat_e7, lon_e7, alt_mm,
             UNKNOWN_U16, UNKNOWN_U16, mo["vel"], mo["cog"],
             p["satellites_visible"]]
        )

    write_csv(out_dir / f"{mission_id}_GLOBAL_POSITION_INT.csv", "GLOBAL_POSITION_INT", gpi_rows)
    write_csv(out_dir / f"{mission_id}_GPS_RAW_INT.csv", "GPS_RAW_INT", gps_rows)
    write_placeholders(out_dir, mission_id)


# ---------------------------------------------------------------------------
# Source parsers
# ---------------------------------------------------------------------------

def load_vehicleout(path: Path) -> list[dict]:
    """*_vehicleOut.txt: idx, lon, lat, alt_m, <unused>, timestamp, fix_type,
    satellites_visible. No home altitude is recorded for this run, so
    alt is treated as already relative-to-home (it starts near 0 m)."""
    points = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, lon, lat, alt, _unused, ts, fix_type, sats = line.split(",")
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").timestamp()
            alt_m = float(alt)
            points.append(
                {
                    "t_epoch_s": t,
                    "lat": float(lat),
                    "lon": float(lon),
                    "alt_m": alt_m,
                    "relative_alt_m": alt_m,
                    "fix_type": int(fix_type),
                    "satellites_visible": int(sats),
                    "hdg_cdeg": None,
                }
            )
    return points


def load_telemetry_json(path: Path) -> list[dict]:
    points = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)["telemetry"]
            lat, lon, alt_m, ts = rec["position"]
            home_alt_m = rec["home"][2]
            points.append(
                {
                    "t_epoch_s": parse_iso(ts),
                    "lat": lat,
                    "lon": lon,
                    "alt_m": alt_m,
                    "relative_alt_m": alt_m - home_alt_m,
                    "fix_type": rec["gps"]["fix_type"],
                    "satellites_visible": rec["gps"]["satellites_visible"],
                    "hdg_cdeg": round(rec["heading"] * 100),
                }
            )
    return points


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_by_duration(points: list[dict], chunk_seconds: float) -> list[list[dict]]:
    """Split a continuous recording into several missions so that
    split_missions() (mission-level, no window leakage) has more than one
    mission to split. Chunk boundaries do not affect leakage since chunks
    become independent missions, each assigned wholly to one split."""
    if not points:
        return []
    chunks, current, chunk_start = [], [], points[0]["t_epoch_s"]
    for p in points:
        if p["t_epoch_s"] - chunk_start >= chunk_seconds and current:
            chunks.append(current)
            current, chunk_start = [], p["t_epoch_s"]
        current.append(p)
    if current:
        # Merge a short trailing remainder into the previous chunk instead
        # of emitting a near-empty mission.
        if chunks and len(current) < 0.34 * chunk_seconds:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/testing/flypaw"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--chunk-seconds", type=float, default=180.0,
        help="Max duration per mission when chunking the dense 2022-01-13 run",
    )
    parser.add_argument(
        "--start-index", type=int, default=1,
        help="First mission number (mission_%%05d)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or (input_dir / "telemetry")
    output_dir.mkdir(parents=True, exist_ok=True)

    mission_idx = args.start_index
    written = []

    vehicleout_files = sorted(input_dir.glob("*_vehicleOut.txt"))
    for vf in vehicleout_files:
        points = load_vehicleout(vf)
        for chunk in chunk_by_duration(points, args.chunk_seconds):
            mission_id = f"mission_{mission_idx:05d}"
            write_mission(output_dir, mission_id, chunk)
            written.append((mission_id, vf.name, len(chunk)))
            mission_idx += 1

    telemetry_files = sorted(input_dir.glob("telemetry_*.json"))
    for tf in telemetry_files:
        points = load_telemetry_json(tf)
        if not points:
            continue
        mission_id = f"mission_{mission_idx:05d}"
        write_mission(output_dir, mission_id, points)
        written.append((mission_id, tf.name, len(points)))
        mission_idx += 1

    print(f"Wrote {len(written)} missions to {output_dir}/")
    for mission_id, source, n in written:
        print(f"  {mission_id}  <- {source}  ({n} points)")
    if not written:
        print("No source files found (expected *_vehicleOut.txt / telemetry_*.json)")


if __name__ == "__main__":
    main()
