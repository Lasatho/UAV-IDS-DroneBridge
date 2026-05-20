#!/usr/bin/env python3
"""
validate_missions.py — Post-hoc validation of completed missions.

Compares recorded telemetry (GLOBAL_POSITION_INT) against planned waypoints
to verify each mission was actually flown as intended.

Checks per mission:
  1. Takeoff detected (altitude > threshold)
  2. Each nav waypoint was reached (within acceptance radius)
  3. Landing detected (altitude returns to ~0)
  4. Mission duration within expected bounds
  5. No prolonged GPS loss or telemetry gaps

Reads from:
  - dataset/waypoints/<mission_id>.waypoints
  - dataset/telemetry/<mission_id>_GLOBAL_POSITION_INT.csv
  - dataset/missions_manifest.csv

Outputs:
  - dataset/validation_report.csv
  - Per-mission pass/fail with details

Usage:
    python3 validate_missions.py --dataset ./dataset
    python3 validate_missions.py --dataset ./dataset --mission mission_00001
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS = 6378137.0

# Validation thresholds
TAKEOFF_ALT_THRESHOLD = 3.0      # meters — must exceed this to count as airborne
LANDING_ALT_THRESHOLD = 2.0      # meters — must drop below this to count as landed
WP_ACCEPT_RADIUS = 15.0          # meters — how close the drone must pass to a WP
MAX_TELEMETRY_GAP_S = 5.0        # seconds — max gap between telemetry messages
MIN_DURATION_S = 30.0            # minimum plausible mission duration
MAX_DURATION_S = 1800.0          # 30 min — maximum plausible mission duration

# MAVLink commands that represent navigable waypoints
NAV_COMMANDS = {16, 17, 18, 19, 22}  # WP, LOITER_UNLIM, LOITER_TURNS, LOITER_TIME, TAKEOFF


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class PlannedWaypoint:
    index: int
    command: int
    lat: float
    lon: float
    alt: float


@dataclass
class TelemetryPoint:
    timestamp: float  # seconds since epoch or relative
    lat: float
    lon: float
    alt_rel: float    # relative altitude in meters


@dataclass
class ValidationResult:
    mission_id: str
    status: str = "unknown"          # pass, fail, skipped
    takeoff_ok: bool = False
    landing_ok: bool = False
    waypoints_total: int = 0
    waypoints_reached: int = 0
    waypoints_missed: List[int] = field(default_factory=list)
    duration_s: float = 0.0
    duration_ok: bool = False
    max_telemetry_gap_s: float = 0.0
    telemetry_gap_ok: bool = False
    telemetry_points: int = 0
    details: str = ""

    @property
    def all_waypoints_reached(self) -> bool:
        return self.waypoints_reached == self.waypoints_total

    def evaluate(self):
        """Set overall pass/fail based on individual checks."""
        checks = [
            self.takeoff_ok,
            self.landing_ok,
            self.all_waypoints_reached,
            self.duration_ok,
            self.telemetry_gap_ok,
        ]
        self.status = "pass" if all(checks) else "fail"


def load_waypoints(wp_file: Path) -> List[PlannedWaypoint]:
    """Parse QGC WPL 110 waypoint file."""
    waypoints = []
    with open(wp_file, "r") as f:
        header = f.readline().strip()
        if header != "QGC WPL 110":
            raise ValueError(f"Bad header: {header}")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            wp = PlannedWaypoint(
                index=int(parts[0]),
                command=int(parts[3]),
                lat=float(parts[8]),
                lon=float(parts[9]),
                alt=float(parts[10]),
            )
            waypoints.append(wp)
    return waypoints


def load_telemetry(telem_file: Path) -> List[TelemetryPoint]:
    """
    Load GLOBAL_POSITION_INT telemetry CSV.
    Expected columns: timestamp, lat, lon, alt, relative_alt, vx, vy, vz, hdg
    lat/lon are in degE7, relative_alt in mm (standard MAVLink encoding).
    """
    points = []
    with open(telem_file, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames

        # Auto-detect column format
        # pymavlink telemetry_logger may use different column names
        # Try to be flexible
        for row in reader:
            try:
                ts = float(row.get("timestamp", row.get("time_boot_ms", 0)))

                # Handle both degE7 (integer) and decimal degree formats
                raw_lat = float(row.get("lat", 0))
                raw_lon = float(row.get("lon", 0))
                if abs(raw_lat) > 900:
                    # degE7 format
                    lat = raw_lat / 1e7
                    lon = raw_lon / 1e7
                else:
                    lat = raw_lat
                    lon = raw_lon

                # Handle both mm and meter formats for altitude
                raw_alt = float(row.get("relative_alt", row.get("alt_rel", 0)))
                if abs(raw_alt) > 10000:
                    # mm format
                    alt_rel = raw_alt / 1000.0
                else:
                    alt_rel = raw_alt

                points.append(TelemetryPoint(
                    timestamp=ts, lat=lat, lon=lon, alt_rel=alt_rel
                ))
            except (ValueError, KeyError) as e:
                continue

    return points


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_mission(
    mission_id: str,
    waypoints: List[PlannedWaypoint],
    telemetry: List[TelemetryPoint],
    accept_radius: float = WP_ACCEPT_RADIUS,
) -> ValidationResult:
    """Run all validation checks for a single mission."""

    result = ValidationResult(mission_id=mission_id)
    result.telemetry_points = len(telemetry)

    if len(telemetry) < 10:
        result.status = "fail"
        result.details = f"Insufficient telemetry: {len(telemetry)} points"
        return result

    # --- Duration check ---
    t_start = telemetry[0].timestamp
    t_end = telemetry[-1].timestamp
    result.duration_s = t_end - t_start
    result.duration_ok = MIN_DURATION_S <= result.duration_s <= MAX_DURATION_S

    # --- Telemetry gap check ---
    max_gap = 0.0
    for i in range(1, len(telemetry)):
        gap = telemetry[i].timestamp - telemetry[i - 1].timestamp
        if gap > max_gap:
            max_gap = gap
    result.max_telemetry_gap_s = max_gap
    result.telemetry_gap_ok = max_gap <= MAX_TELEMETRY_GAP_S

    # --- Takeoff check ---
    max_alt = max(p.alt_rel for p in telemetry)
    result.takeoff_ok = max_alt > TAKEOFF_ALT_THRESHOLD

    # --- Landing check ---
    # Check last 10% of telemetry for altitude drop
    tail_start = int(len(telemetry) * 0.9)
    tail = telemetry[tail_start:]
    min_tail_alt = min(p.alt_rel for p in tail) if tail else 999
    result.landing_ok = min_tail_alt < LANDING_ALT_THRESHOLD

    # --- Waypoint reach check ---
    # Only check navigable waypoints (skip home at index 0 and RTL)
    nav_wps = [
        wp for wp in waypoints
        if wp.command in NAV_COMMANDS and wp.index > 0 and wp.lat != 0.0
    ]
    result.waypoints_total = len(nav_wps)

    for wp in nav_wps:
        reached = False
        for tp in telemetry:
            dist = haversine(wp.lat, wp.lon, tp.lat, tp.lon)
            if dist <= accept_radius:
                reached = True
                break
        if reached:
            result.waypoints_reached += 1
        else:
            result.waypoints_missed.append(wp.index)

    # --- Build details string ---
    issues = []
    if not result.takeoff_ok:
        issues.append(f"no takeoff (max alt={max_alt:.1f}m)")
    if not result.landing_ok:
        issues.append(f"no landing (min tail alt={min_tail_alt:.1f}m)")
    if not result.all_waypoints_reached:
        issues.append(f"missed WPs: {result.waypoints_missed}")
    if not result.duration_ok:
        issues.append(f"duration {result.duration_s:.0f}s out of bounds")
    if not result.telemetry_gap_ok:
        issues.append(f"telemetry gap {result.max_telemetry_gap_s:.1f}s")
    result.details = "; ".join(issues) if issues else "all checks passed"

    result.evaluate()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_telemetry_file(telem_dir: Path, mission_id: str) -> Optional[Path]:
    """Find the GLOBAL_POSITION_INT telemetry file for a mission."""
    # Try common naming patterns
    candidates = [
        telem_dir / f"{mission_id}_GLOBAL_POSITION_INT.csv",
        telem_dir / f"run_{mission_id}_GLOBAL_POSITION_INT.csv",
        telem_dir / f"{mission_id}.csv",
        telem_dir / f"{mission_id}_gpi.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Glob fallback
    matches = list(telem_dir.glob(f"{mission_id}*GLOBAL*"))
    if matches:
        return matches[0]
    matches = list(telem_dir.glob(f"{mission_id}*.csv"))
    if matches:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc validation of completed missions"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--mission", type=str, default=None,
        help="Validate single mission by ID (default: all completed)"
    )
    parser.add_argument(
        "--accept-radius", type=float, default=WP_ACCEPT_RADIUS,
        help=f"Waypoint acceptance radius in meters (default: {WP_ACCEPT_RADIUS})"
    )
    args = parser.parse_args()

    dataset = args.dataset
    wp_dir = dataset / "waypoints"
    telem_dir = dataset / "telemetry"
    manifest_path = dataset / "missions_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # Determine which missions to validate
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missions = list(reader)

    if args.mission:
        missions = [m for m in missions if m["mission_id"] == args.mission]
        if not missions:
            print(f"Mission {args.mission} not found in manifest.")
            sys.exit(1)
    else:
        missions = [m for m in missions if m["status"] == "completed"]

    if not missions:
        print("No completed missions to validate.")
        sys.exit(0)

    print(f"Validating {len(missions)} missions...")

    # Run validation
    results = []
    for m in missions:
        mid = m["mission_id"]

        # Load waypoints
        wp_file = wp_dir / m.get("waypoint_file", f"{mid}.waypoints")
        if not wp_file.exists():
            results.append(ValidationResult(
                mission_id=mid, status="skipped",
                details=f"Waypoint file not found: {wp_file.name}"
            ))
            continue

        # Find telemetry
        telem_file = find_telemetry_file(telem_dir, mid)
        if telem_file is None:
            results.append(ValidationResult(
                mission_id=mid, status="skipped",
                details="No telemetry file found"
            ))
            continue

        # Load and validate
        try:
            waypoints = load_waypoints(wp_file)
            telemetry = load_telemetry(telem_file)
            result = validate_mission(mid, waypoints, telemetry, args.accept_radius)
            results.append(result)
        except Exception as e:
            results.append(ValidationResult(
                mission_id=mid, status="fail",
                details=f"Error: {e}"
            ))

    # Write report
    report_path = dataset / "validation_report.csv"
    report_fields = [
        "mission_id", "status", "takeoff_ok", "landing_ok",
        "waypoints_total", "waypoints_reached", "waypoints_missed",
        "duration_s", "duration_ok", "max_telemetry_gap_s",
        "telemetry_gap_ok", "telemetry_points", "details"
    ]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "mission_id": r.mission_id,
                "status": r.status,
                "takeoff_ok": r.takeoff_ok,
                "landing_ok": r.landing_ok,
                "waypoints_total": r.waypoints_total,
                "waypoints_reached": r.waypoints_reached,
                "waypoints_missed": r.waypoints_missed if r.waypoints_missed else "",
                "duration_s": f"{r.duration_s:.1f}",
                "duration_ok": r.duration_ok,
                "max_telemetry_gap_s": f"{r.max_telemetry_gap_s:.2f}",
                "telemetry_gap_ok": r.telemetry_gap_ok,
                "telemetry_points": r.telemetry_points,
                "details": r.details,
            })

    # Summary
    n_pass = sum(1 for r in results if r.status == "pass")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skipped")

    print(f"\n{'='*50}")
    print(f"Validation Report: {report_path}")
    print(f"{'='*50}")
    print(f"  Total:   {len(results)}")
    print(f"  Pass:    {n_pass}")
    print(f"  Fail:    {n_fail}")
    print(f"  Skipped: {n_skip}")

    if n_fail > 0:
        print(f"\nFailed missions:")
        for r in results:
            if r.status == "fail":
                print(f"  {r.mission_id}: {r.details}")

    print(f"\nPass rate: {n_pass}/{n_pass + n_fail} "
          f"({100 * n_pass / max(1, n_pass + n_fail):.1f}%)")


if __name__ == "__main__":
    main()