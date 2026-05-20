#!/usr/bin/env python3
"""
Waypoint Generator for IDS Normal-Flight Dataset v1
Full Factorial Design over all parameter dimensions.

Generates:
  - .waypoints files (QGC WPL 110 format) for ArduPilot SITL
  - missions_manifest.csv with all parameter combinations
  - meta/<mission_id>_params.json per mission

Profiles: Rectangle, Figure-8, Star, Zigzag Climb/Descent, Loiter Circles
Drone Presets: Conservative, Standard, Dynamic, Fast
Wind Profiles: Calm, Light, Moderate, Gusty
Wind Directions: N, NE, E, SE (0°, 45°, 90°, 135°)

Usage:
    python3 generate_missions.py --output-dir ./dataset
    python3 generate_missions.py --output-dir ./dataset --dry-run  # only print count
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Home position (arbitrary, flat Gazebo world — only relative offsets matter)
HOME_LAT = -35.363261
HOME_LNG = 149.165230
HOME_ALT = 0.0

# Earth radius for offset calculations (meters)
EARTH_RADIUS = 6378137.0

# MAVLink commands
CMD_WAYPOINT = 16       # MAV_CMD_NAV_WAYPOINT
CMD_TAKEOFF = 22        # MAV_CMD_NAV_TAKEOFF
CMD_LOITER_UNLIM = 17   # MAV_CMD_NAV_LOITER_UNLIM
CMD_LOITER_TURNS = 18   # MAV_CMD_NAV_LOITER_TURNS
CMD_LOITER_TIME = 19    # MAV_CMD_NAV_LOITER_TIME
CMD_RTL = 20            # MAV_CMD_NAV_RETURN_TO_LAUNCH
CMD_LAND = 21           # MAV_CMD_NAV_LAND

# Frame: MAV_FRAME_GLOBAL_RELATIVE_ALT
FRAME_REL_ALT = 3

# Validation limits
ALT_MIN = 10
ALT_MAX = 120
WP_MIN_DISTANCE = 20
GEOFENCE_RADIUS = 3000
SPEED_MIN = 200
SPEED_MAX = 1500


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def offset_meters_to_latlon(
    lat: float, lon: float, north_m: float, east_m: float
) -> Tuple[float, float]:
    """Offset a lat/lon position by meters north and east."""
    d_lat = north_m / EARTH_RADIUS
    d_lon = east_m / (EARTH_RADIUS * math.cos(math.radians(lat)))
    return lat + math.degrees(d_lat), lon + math.degrees(d_lon)


def distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Haversine distance between two lat/lon points in meters."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Waypoint data structures
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    index: int
    current: int  # 1 for home (index 0), 0 otherwise
    frame: int
    command: int
    p1: float = 0.0  # hold time / loiter time / turns
    p2: float = 0.0  # acceptance radius
    p3: float = 0.0  # pass-through (0=stop, 1=flythrough) / loiter radius
    p4: float = 0.0  # yaw
    lat: float = 0.0
    lng: float = 0.0
    alt: float = 0.0
    autocontinue: int = 1

    def to_line(self) -> str:
        return (
            f"{self.index}\t{self.current}\t{self.frame}\t{self.command}\t"
            f"{self.p1}\t{self.p2}\t{self.p3}\t{self.p4}\t"
            f"{self.lat:.7f}\t{self.lng:.7f}\t{self.alt:.2f}\t{self.autocontinue}"
        )


def write_waypoints(filepath: Path, waypoints: List[Waypoint]):
    """Write a list of Waypoints to QGC WPL 110 format."""
    with open(filepath, "w") as f:
        f.write("QGC WPL 110\n")
        for wp in waypoints:
            f.write(wp.to_line() + "\n")


def make_home() -> Waypoint:
    return Waypoint(
        index=0, current=1, frame=0, command=CMD_WAYPOINT,
        lat=HOME_LAT, lng=HOME_LNG, alt=HOME_ALT
    )


def make_takeoff(index: int, alt: float) -> Waypoint:
    return Waypoint(
        index=index, current=0, frame=FRAME_REL_ALT, command=CMD_TAKEOFF,
        lat=HOME_LAT, lng=HOME_LNG, alt=alt
    )


def make_wp(index: int, lat: float, lng: float, alt: float) -> Waypoint:
    return Waypoint(
        index=index, current=0, frame=FRAME_REL_ALT, command=CMD_WAYPOINT,
        lat=lat, lng=lng, alt=alt
    )


def make_rtl(index: int) -> Waypoint:
    return Waypoint(
        index=index, current=0, frame=FRAME_REL_ALT, command=CMD_RTL
    )


def make_loiter_time(
    index: int, lat: float, lng: float, alt: float,
    time_s: float, radius: float = 0.0, direction: int = 1
) -> Waypoint:
    """
    LOITER_TIME: p1=time(s), p3=radius (positive=CW, negative=CCW, 0=hover).
    direction: 1=CW, -1=CCW
    """
    r = abs(radius) * direction
    return Waypoint(
        index=index, current=0, frame=FRAME_REL_ALT, command=CMD_LOITER_TIME,
        p1=time_s, p3=r,
        lat=lat, lng=lng, alt=alt
    )


# ---------------------------------------------------------------------------
# Profile generators
# ---------------------------------------------------------------------------

def generate_rectangle(
    side_m: float, alt: float, laps: int
) -> List[Waypoint]:
    """Rectangle pattern: 4 corners, repeated `laps` times."""
    wps = [make_home(), make_takeoff(1, alt)]
    half = side_m / 2.0
    corners = [
        offset_meters_to_latlon(HOME_LAT, HOME_LNG, half, half),
        offset_meters_to_latlon(HOME_LAT, HOME_LNG, half, -half),
        offset_meters_to_latlon(HOME_LAT, HOME_LNG, -half, -half),
        offset_meters_to_latlon(HOME_LAT, HOME_LNG, -half, half),
    ]
    idx = 2
    for _ in range(laps):
        for clat, clng in corners:
            wps.append(make_wp(idx, clat, clng, alt))
            idx += 1
    wps.append(make_rtl(idx))
    return wps


def generate_figure8(
    radius_m: float, alt: float, alt_delta: float = 0.0
) -> List[Waypoint]:
    """
    Figure-8 (lemniscate approximation): two tangential circles.
    alt_delta: if > 0, second loop is at alt + alt_delta.
    """
    wps = [make_home(), make_takeoff(1, alt)]
    n_points = 8  # per circle
    idx = 2

    # Circle 1: centered north of home
    c1_lat, c1_lng = offset_meters_to_latlon(HOME_LAT, HOME_LNG, radius_m, 0)
    alt1 = alt
    for i in range(n_points):
        angle = 2.0 * math.pi * i / n_points
        n = radius_m * math.cos(angle)
        e = radius_m * math.sin(angle)
        wlat, wlng = offset_meters_to_latlon(c1_lat, c1_lng, n, e)
        wps.append(make_wp(idx, wlat, wlng, alt1))
        idx += 1

    # Circle 2: centered south of home
    c2_lat, c2_lng = offset_meters_to_latlon(HOME_LAT, HOME_LNG, -radius_m, 0)
    alt2 = alt + alt_delta
    for i in range(n_points):
        # Reverse direction for smooth figure-8 transition
        angle = -2.0 * math.pi * i / n_points
        n = radius_m * math.cos(angle)
        e = radius_m * math.sin(angle)
        wlat, wlng = offset_meters_to_latlon(c2_lat, c2_lng, n, e)
        wps.append(make_wp(idx, wlat, wlng, alt2))
        idx += 1

    wps.append(make_rtl(idx))
    return wps


def generate_star(
    radius_m: float, n_tips: int, alt: float
) -> List[Waypoint]:
    """Star/radial pattern: center → tip → center, repeated for each tip."""
    wps = [make_home(), make_takeoff(1, alt)]
    idx = 2

    for i in range(n_tips):
        angle = 2.0 * math.pi * i / n_tips
        n = radius_m * math.cos(angle)
        e = radius_m * math.sin(angle)
        tip_lat, tip_lng = offset_meters_to_latlon(HOME_LAT, HOME_LNG, n, e)
        # Fly to tip
        wps.append(make_wp(idx, tip_lat, tip_lng, alt))
        idx += 1
        # Return to center
        wps.append(make_wp(idx, HOME_LAT, HOME_LNG, alt))
        idx += 1

    wps.append(make_rtl(idx))
    return wps


def generate_zigzag(
    alt_delta: float, h_distance: float,
    alt_low: float, alt_high: float
) -> List[Waypoint]:
    """
    Zigzag climb/descent: alternating altitudes along a line.
    Generates enough segments to span the altitude band at least once up and down.
    """
    wps = [make_home(), make_takeoff(1, alt_low)]
    idx = 2
    east_offset = 0.0
    current_alt = alt_low
    going_up = True
 
    # Generate segments until we've gone up and come back down
    # Cap at geofence: east_offset = segments * h_distance must stay < GEOFENCE_RADIUS
    segments = 0
    max_by_alt = max(4, int(2 * (alt_high - alt_low) / alt_delta) + 2)
    max_by_geofence = int(GEOFENCE_RADIUS / h_distance) - 1
    max_segments = min(max_by_alt, max_by_geofence)
 
    for seg in range(max_segments):
        east_offset += h_distance
        if going_up:
            current_alt += alt_delta
            if current_alt >= alt_high:
                current_alt = alt_high
                going_up = False
        else:
            current_alt -= alt_delta
            if current_alt <= alt_low:
                current_alt = alt_low
                going_up = True
 
        wlat, wlng = offset_meters_to_latlon(HOME_LAT, HOME_LNG, 0, east_offset)
        # Alternate north/south for zigzag lateral movement
        n_offset = h_distance / 2.0 * (1 if seg % 2 == 0 else -1)
        wlat, wlng = offset_meters_to_latlon(HOME_LAT, HOME_LNG, n_offset, east_offset)
        wps.append(make_wp(idx, wlat, wlng, current_alt))
        idx += 1
        segments += 1
 
    wps.append(make_rtl(idx))
    return wps


def generate_loiter(
    positions: int, loiter_radius: float, loiter_time_s: float,
    alt: float, direction: int
) -> List[Waypoint]:
    """
    Loiter/hold pattern: fly to N positions, loiter at each.
    positions: 2 or 3 loiter points spread around home.
    direction: 1=CW, -1=CCW
    """
    wps = [make_home(), make_takeoff(1, alt)]
    idx = 2
    spread = 150.0  # meters between loiter positions

    for i in range(positions):
        angle = 2.0 * math.pi * i / positions
        n = spread * math.cos(angle)
        e = spread * math.sin(angle)
        plat, plng = offset_meters_to_latlon(HOME_LAT, HOME_LNG, n, e)
        # Fly to position
        wps.append(make_wp(idx, plat, plng, alt))
        idx += 1
        # Loiter
        wps.append(make_loiter_time(idx, plat, plng, alt, loiter_time_s, loiter_radius, direction))
        idx += 1

    wps.append(make_rtl(idx))
    return wps


# ---------------------------------------------------------------------------
# Parameter space definitions (from concept doc)
# ---------------------------------------------------------------------------

# Profile geometry variants
RECTANGLE_PARAMS = list(product(
    [100, 200, 400, 800],   # side_m
    [15, 30, 60, 100],      # alt
    [1, 2, 3],              # laps
))

FIGURE8_PARAMS = (
    # Constant altitude
    [(r, a, 0) for r, a in product(
        [50, 100, 200],     # radius
        [20, 50, 80],       # alt
    )]
    # With altitude change between loops
    + [(r, 40, 15) for r in [50, 100, 200]]
)

STAR_PARAMS = list(product(
    [100, 250, 500],        # radius
    [5, 6, 8],              # n_tips
    [20, 50, 80],           # alt
))

ZIGZAG_PARAMS = list(product(
    [10, 20, 40],           # alt_delta
    [50, 100, 200],         # h_distance
    [(15, 60), (30, 100), (15, 120)],  # alt_band (low, high)
))

LOITER_PARAMS = list(product(
    [0, 20, 50],            # loiter_radius (0 = hover)
    [60, 120, 180],         # loiter_time_s
    [15, 30, 60],           # alt
    [1, -1],                # direction (CW, CCW)
))

# Drone presets
DRONE_PRESETS = {
    "conservative": {
        "WPNAV_SPEED": 300,
        "WPNAV_SPEED_UP": 150,
        "WPNAV_SPEED_DN": 100,
        "WPNAV_LOIT_SPEED": 250,
        "WPNAV_RADIUS": 200,
        "RTL_ALT": 3000,
    },
    "standard": {
        "WPNAV_SPEED": 500,
        "WPNAV_SPEED_UP": 250,
        "WPNAV_SPEED_DN": 150,
        "WPNAV_LOIT_SPEED": 500,
        "WPNAV_RADIUS": 200,
        "RTL_ALT": 3000,
    },
    "dynamic": {
        "WPNAV_SPEED": 1000,
        "WPNAV_SPEED_UP": 350,
        "WPNAV_SPEED_DN": 250,
        "WPNAV_LOIT_SPEED": 750,
        "WPNAV_RADIUS": 100,
        "RTL_ALT": 3000,
    },
    "fast": {
        "WPNAV_SPEED": 1200,
        "WPNAV_SPEED_UP": 350,
        "WPNAV_SPEED_DN": 250,
        "WPNAV_LOIT_SPEED": 750,
        "WPNAV_RADIUS": 500,
        "RTL_ALT": 6000,
    },
}

# Wind profiles
WIND_PROFILES = {
    "calm": {"speed_min": 0.0, "speed_max": 1.0, "turbulence": "none"},
    "light": {"speed_min": 2.0, "speed_max": 4.0, "turbulence": "low"},
    "moderate": {"speed_min": 5.0, "speed_max": 8.0, "turbulence": "medium"},
    "gusty": {"speed_min": 3.0, "speed_max": 10.0, "turbulence": "high"},
}

# Wind directions (azimuth degrees: 0=N, 45=NE, 90=E, 135=SE)
WIND_DIRECTIONS = [0, 45, 90, 135]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_mission(wps: List[Waypoint]) -> Tuple[bool, str]:
    """Validate a mission against quality criteria from concept doc."""
    nav_wps = [w for w in wps if w.command in (CMD_WAYPOINT, CMD_TAKEOFF, CMD_LOITER_TIME) and w.index > 0]

    # Altitude check
    for w in nav_wps:
        if w.alt < ALT_MIN:
            return False, f"WP {w.index}: alt {w.alt}m < {ALT_MIN}m"
        if w.alt > ALT_MAX:
            return False, f"WP {w.index}: alt {w.alt}m > {ALT_MAX}m"

    # Distance check (between consecutive nav waypoints)
    for i in range(1, len(nav_wps)):
        d = distance_meters(nav_wps[i-1].lat, nav_wps[i-1].lng,
                           nav_wps[i].lat, nav_wps[i].lng)
        # Skip distance check for loiter points at same position as preceding WP
        if d < WP_MIN_DISTANCE and d > 0.1:
            return False, f"WP {nav_wps[i].index}: distance {d:.1f}m < {WP_MIN_DISTANCE}m"

    # Geofence check
    for w in nav_wps:
        d = distance_meters(HOME_LAT, HOME_LNG, w.lat, w.lng)
        if d > GEOFENCE_RADIUS:
            return False, f"WP {w.index}: {d:.0f}m from home > {GEOFENCE_RADIUS}m geofence"

    # Structure check
    if wps[1].command != CMD_TAKEOFF:
        return False, "First mission WP (index 1) must be TAKEOFF"
    if wps[-1].command != CMD_RTL:
        return False, "Last WP must be RTL"

    return True, "OK"


# ---------------------------------------------------------------------------
# Mission generation dispatcher
# ---------------------------------------------------------------------------

def generate_profile_waypoints(
    profile: str, geometry_params: tuple
) -> List[Waypoint]:
    """Dispatch to the correct profile generator."""
    if profile == "rectangle":
        side_m, alt, laps = geometry_params
        return generate_rectangle(side_m, alt, laps)
    elif profile == "figure8":
        radius, alt, alt_delta = geometry_params
        return generate_figure8(radius, alt, alt_delta)
    elif profile == "star":
        radius, n_tips, alt = geometry_params
        return generate_star(radius, n_tips, alt)
    elif profile == "zigzag":
        alt_delta, h_distance, (alt_low, alt_high) = geometry_params
        return generate_zigzag(alt_delta, h_distance, alt_low, alt_high)
    elif profile == "loiter":
        loiter_radius, loiter_time_s, alt, direction = geometry_params
        return generate_loiter(3, loiter_radius, loiter_time_s, alt, direction)
    else:
        raise ValueError(f"Unknown profile: {profile}")


# ---------------------------------------------------------------------------
# Full factorial enumeration
# ---------------------------------------------------------------------------

PROFILES = {
    "rectangle": RECTANGLE_PARAMS,
    "figure8": FIGURE8_PARAMS,
    "star": STAR_PARAMS,
    "zigzag": ZIGZAG_PARAMS,
    "loiter": LOITER_PARAMS,
}


def enumerate_all_missions():
    """
    Full factorial: profile × geometry × drone_preset × wind_profile × wind_direction.
    Yields (profile, geometry_params, drone_preset_name, wind_profile_name, wind_direction).
    """
    for profile_name, geo_variants in PROFILES.items():
        for geo_params in geo_variants:
            for preset_name in DRONE_PRESETS:
                for wind_name in WIND_PROFILES:
                    for wind_dir in WIND_DIRECTIONS:
                        yield (profile_name, geo_params, preset_name, wind_name, wind_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate IDS Normal-Flight Dataset v1")
    parser.add_argument("--output-dir", type=str, default="./dataset",
                        help="Root output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count missions, don't write files")
    args = parser.parse_args()

    # Count per profile
    print("=== Mission Count per Profile ===")
    total_geo = 0
    for pname, gvariants in PROFILES.items():
        n_geo = len(gvariants)
        n_full = n_geo * len(DRONE_PRESETS) * len(WIND_PROFILES) * len(WIND_DIRECTIONS)
        print(f"  {pname:12s}: {n_geo:4d} geometries × "
              f"{len(DRONE_PRESETS)} presets × {len(WIND_PROFILES)} wind × "
              f"{len(WIND_DIRECTIONS)} dirs = {n_full:6d} missions")
        total_geo += n_geo

    total = total_geo * len(DRONE_PRESETS) * len(WIND_PROFILES) * len(WIND_DIRECTIONS)
    print(f"\n  Total geometries: {total_geo}")
    print(f"  Total missions (full factorial): {total}")

    if args.dry_run:
        sys.exit(0)

    # Create directory structure
    out = Path(args.output_dir)
    (out / "waypoints").mkdir(parents=True, exist_ok=True)
    (out / "pcap").mkdir(exist_ok=True)
    (out / "telemetry").mkdir(exist_ok=True)
    (out / "phase_labels").mkdir(exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)

    # Generate all missions
    manifest_path = out / "missions_manifest.csv"
    manifest_fields = [
        "mission_id", "profile", "geometry_params", "drone_preset",
        "wind_profile", "wind_direction_deg", "wind_speed_min", "wind_speed_max",
        "waypoint_file", "n_waypoints", "validation", "status"
    ]

    generated = 0
    skipped = 0

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=manifest_fields)
        writer.writeheader()

        for mission_idx, (profile, geo, preset, wind, wind_dir) in enumerate(
            enumerate_all_missions(), start=1
        ):
            mission_id = f"mission_{mission_idx:05d}"
            wp_filename = f"{mission_id}.waypoints"
            wp_path = out / "waypoints" / wp_filename

            # Generate waypoints
            wps = generate_profile_waypoints(profile, geo)

            # Validate
            valid, reason = validate_mission(wps)

            wind_info = WIND_PROFILES[wind]

            row = {
                "mission_id": mission_id,
                "profile": profile,
                "geometry_params": json.dumps(geo if not isinstance(geo[-1], tuple) else list(geo[:-1]) + list(geo[-1])),
                "drone_preset": preset,
                "wind_profile": wind,
                "wind_direction_deg": wind_dir,
                "wind_speed_min": wind_info["speed_min"],
                "wind_speed_max": wind_info["speed_max"],
                "waypoint_file": wp_filename,
                "n_waypoints": len(wps),
                "validation": reason,
                "status": "pending" if valid else "invalid",
            }
            writer.writeheader() if mission_idx == 1 else None
            writer.writerow(row)

            if valid:
                write_waypoints(wp_path, wps)
                # Write meta JSON
                meta = {
                    "mission_id": mission_id,
                    "profile": profile,
                    "geometry_params": geo if not isinstance(geo[-1], tuple) else list(geo[:-1]) + list(geo[-1]),
                    "drone_preset": preset,
                    "drone_params": DRONE_PRESETS[preset],
                    "wind_profile": wind,
                    "wind_params": wind_info,
                    "wind_direction_deg": wind_dir,
                    "n_waypoints": len(wps),
                    "home": {"lat": HOME_LAT, "lng": HOME_LNG, "alt": HOME_ALT},
                }
                with open(out / "meta" / f"{mission_id}_params.json", "w") as mf:
                    json.dump(meta, mf, indent=2, default=str)
                generated += 1
            else:
                skipped += 1

            # Progress
            if mission_idx % 1000 == 0:
                print(f"  ... {mission_idx} processed ({generated} valid, {skipped} invalid)")

    print(f"\nDone: {generated} missions generated, {skipped} invalid/skipped")
    print(f"Manifest: {manifest_path}")
    print(f"Waypoints: {out / 'waypoints'}/")
    print(f"Meta: {out / 'meta'}/")


if __name__ == "__main__":
    main()