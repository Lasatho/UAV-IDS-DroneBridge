#!/usr/bin/env python3
"""
generate_attack_missions.py — attack dataset v2 on the flight configurations
of the normal dataset v2.

Design: every v2 geometry x drone preset x attack type, one flight each
(123 x 4 x 5 = 2460 missions, ids from --id-offset + 1, default 40001, so
the first attack set mission_2xxxx stays untouched). Waypoints, presets
(ArduCopter 4.8 names, SI units) and wind profiles are those of
generate_missions.py, so normal and attack flights differ only in the attack.

Wind is rotated instead of crossed:
    wind profile    = (geometry + preset + attack) mod 4
    wind direction  = (geometry + attack + 3 * wind profile) mod 4
Each geometry gets every wind profile and direction 5 times; per attack every
wind profile occurs 123 times (a quarter calm) and every wind x direction
cell 30-31 times.

attack_seed = first 32 bits of sha256(mission_id), as in the first attack
set; the orchestrator draws offset and duration from it. c2_hijacking gets
attack_variant land / rtl / terminate in turn (sorted by mission_id).

--split assigns the missions to hosts by estimated wall time: the
split_manifest time model times the measured actual/model ratio per wind
profile and preset (ratios JSON from the v2 run), capped at the 600 s attack
timeout, plus ground time and overhead. Within each attack x wind x preset
stratum the missions go longest-first to the host with the smallest load,
so every host gets the same mix. --pin keeps an attack type on one host
(rf_jamming needs the wmediumd interference model, which only HP runs); the
other strata are then split with shares that balance the total load.

Usage:
    python3 generate_attack_missions.py --output-dir ../dataset \
        [--split hp=0.5 tr=0.5 --ratios ratios_hp.json --pin rf_jamming=hp]
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from generate_missions import (DRONE_PRESETS, HOME_ALT, HOME_LAT, HOME_LNG,
                               PROFILES, WIND_DIRECTIONS, WIND_PROFILES,
                               generate_profile_waypoints, validate_mission,
                               write_waypoints)
from split_manifest import B, GROUND_S, OVERHEAD_S, estimate_s

ATTACKS = ["gnss_spoofing", "gnss_jamming", "rf_jamming",
           "frame_injection", "c2_hijacking"]
VARIANTS = {"c2_hijacking": ("land", "rtl", "terminate")}
ATTACK_TIMEOUT_S = 600

FIELDS = ["mission_id", "profile", "geometry_params", "drone_preset",
          "wind_profile", "wind_direction_deg", "wind_speed_min",
          "wind_speed_max", "attack_type", "waypoint_file", "n_waypoints",
          "validation", "status"]


def attack_seed(mission_id: str) -> int:
    return int(hashlib.sha256(mission_id.encode()).hexdigest()[:8], 16)


def enumerate_attack_missions():
    """Yields (profile, geometry_params, preset, wind profile, direction, attack)."""
    winds = list(WIND_PROFILES)
    geo_idx = 0
    for profile_name, geo_variants in PROFILES.items():
        for geo_params in geo_variants:
            for p_idx, preset_name in enumerate(DRONE_PRESETS):
                for a_idx, attack in enumerate(ATTACKS):
                    w_idx = (geo_idx + p_idx + a_idx) % len(winds)
                    d_idx = (geo_idx + a_idx + 3 * w_idx) % len(WIND_DIRECTIONS)
                    yield (profile_name, geo_params, preset_name, winds[w_idx],
                           WIND_DIRECTIONS[d_idx], attack)
            geo_idx += 1


def estimate_attack_s(wp_path: Path, meta: dict, ratios: dict) -> float:
    """Recorded duration (model x measured ratio, at most ground phase,
    takeoff and the attack timeout) plus the overhead between missions."""
    model = estimate_s(wp_path, meta["drone_params"]) - OVERHEAD_S
    ratio = ratios.get(f"{meta['wind_profile']}|{meta['drone_preset']}", 1.0)
    return min(model * ratio, ATTACK_TIMEOUT_S + GROUND_S + B) + OVERHEAD_S


def split(rows, out_dir: Path, manifest: Path, hosts: dict, ratios: dict,
          pin: dict = None):
    """Pinned attack types go to one host; the shares of the other strata
    are adjusted so that the hosts' total loads follow `hosts`."""
    pin = pin or {}
    est = {}
    for r in rows:
        meta = json.loads((out_dir / "meta" / f"{r['mission_id']}_params.json").read_text())
        est[r["mission_id"]] = estimate_attack_s(
            out_dir / "waypoints" / r["waypoint_file"], meta, ratios)
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["attack_type"], r["wind_profile"], r["drone_preset"])].append(r)
    load = {h: 0.0 for h in hosts}
    assign = {h: [] for h in hosts}
    for r in rows:
        if r["attack_type"] in pin:
            assign[pin[r["attack_type"]]].append(r)
            load[pin[r["attack_type"]]] += est[r["mission_id"]]
    total = sum(est.values())
    free = total - sum(load.values())
    share = {h: max(0.0, hosts[h] / sum(hosts.values()) * total - load[h]) / free
             for h in hosts}
    for key in sorted(k for k in strata if k[0] not in pin):
        part = {h: 0.0 for h in hosts}
        for r in sorted(strata[key], key=lambda r: -est[r["mission_id"]]):
            h = min((h for h in hosts if share[h] > 0),
                    key=lambda h: part[h] / share[h])
            assign[h].append(r)
            part[h] += est[r["mission_id"]]
            load[h] += est[r["mission_id"]]
    stem = manifest.with_suffix("")
    for h, rs in assign.items():
        rs.sort(key=lambda r: r["mission_id"])
        path = Path(f"{stem}_{h}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rs)
        print(f"{h}: {len(rs)} missions, estimated {load[h] / 86400:.1f} days -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate attack dataset v2")
    parser.add_argument("--output-dir", type=Path, default=Path("./dataset"))
    parser.add_argument("--id-offset", type=int, default=40000)
    parser.add_argument("--manifest-name", default="missions_manifest_attacks_v2.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", nargs="+", help="name=share, e.g. hp=0.5 tr=0.5")
    parser.add_argument("--pin", nargs="*", default=[],
                        help="attack=host, e.g. rf_jamming=hp (wmediumd only there)")
    parser.add_argument("--ratios", type=Path,
                        help="actual/model ratios per 'wind|preset' (JSON, "
                             "single value or list -> median)")
    args = parser.parse_args()

    out = args.output_dir
    rows, metas = [], []
    for i, (profile, geo, preset, wind, wind_dir, attack) in enumerate(
            enumerate_attack_missions(), start=1):
        mission_id = f"mission_{args.id_offset + i:05d}"
        wps = generate_profile_waypoints(profile, geo)
        valid, reason = validate_mission(wps)
        geo_out = geo if not isinstance(geo[-1], tuple) else list(geo[:-1]) + list(geo[-1])
        wind_info = WIND_PROFILES[wind]
        rows.append({
            "mission_id": mission_id,
            "profile": profile,
            "geometry_params": json.dumps(geo_out),
            "drone_preset": preset,
            "wind_profile": wind,
            "wind_direction_deg": wind_dir,
            "wind_speed_min": wind_info["speed_min"],
            "wind_speed_max": wind_info["speed_max"],
            "attack_type": attack,
            "waypoint_file": f"{mission_id}.waypoints",
            "n_waypoints": len(wps),
            "validation": reason,
            "status": "pending" if valid else "invalid",
        })
        metas.append((wps, {
            "mission_id": mission_id,
            "profile": profile,
            "geometry_params": geo_out,
            "drone_preset": preset,
            "drone_params": DRONE_PRESETS[preset],
            "wind_profile": wind,
            "wind_params": wind_info,
            "wind_direction_deg": wind_dir,
            "n_waypoints": len(wps),
            "home": {"lat": HOME_LAT, "lng": HOME_LNG, "alt": HOME_ALT},
            "attack_type": attack,
            "attack_seed": attack_seed(mission_id),
        }))

    for attack, variants in VARIANTS.items():
        for j, (_, meta) in enumerate(m for m in metas if m[1]["attack_type"] == attack):
            meta["attack_variant"] = variants[j % len(variants)]

    counts = collections.Counter((r["attack_type"], r["wind_profile"]) for r in rows)
    print(f"{len(rows)} missions, {sum(r['status'] == 'invalid' for r in rows)} invalid")
    for attack in ATTACKS:
        print(f"  {attack:16s}", {w: counts[(attack, w)] for w in WIND_PROFILES})
    if args.dry_run:
        return

    (out / "waypoints").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    for row, (wps, meta) in zip(rows, metas):
        if row["status"] == "invalid":
            continue
        write_waypoints(out / "waypoints" / row["waypoint_file"], wps)
        with open(out / "meta" / f"{row['mission_id']}_params.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    manifest = out / args.manifest_name
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {manifest}")

    if args.split:
        hosts = {h.split("=")[0]: float(h.split("=")[1]) for h in args.split}
        ratios = {}
        if args.ratios:
            # ratios.py output: list of actual/model ratios per wind|preset
            for k, v in json.loads(args.ratios.read_text()).items():
                ratios[k] = v if isinstance(v, float) else sorted(v)[len(v) // 2]
        pin = dict(p.split("=") for p in args.pin)
        split([r for r in rows if r["status"] == "pending"], out, manifest,
              hosts, ratios, pin)


if __name__ == "__main__":
    main()
