#!/usr/bin/env python3
"""Read-only: plausibility of the simulated sensor values (all training features).

Per completed mission: NaN, physical ranges, frozen values in flight, sample
rate, cross-stream consistency (GPS vs. GLOBAL_POSITION_INT, baro vs.
relative_alt) and summary statistics for the outlier/host comparison.
Usage: data_check.py <repo> <manifest> <out.json>
"""
import csv, json, sys
from pathlib import Path
import numpy as np

STREAMS = {
    "GLOBAL_POSITION_INT": ["lat", "lon", "alt", "relative_alt", "vx", "vy", "vz", "hdg"],
    "GPS_RAW_INT": ["fix_type", "lat", "lon", "alt", "eph", "epv", "vel", "cog", "satellites_visible"],
    "RAW_IMU": ["xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro", "xmag", "ymag", "zmag"],
    "SCALED_PRESSURE": ["press_abs", "press_diff", "temperature"],
    "SERVO_OUTPUT_RAW": [f"servo{i}_raw" for i in range(1, 9)],
}
FLIGHT_ALT_M = 1.0
FROZEN_S = 5.0          # identical consecutive feature rows in flight
R_EARTH = 6371000.0


def load(path, cols):
    with open(path) as f:
        rd = csv.DictReader(f)
        rows = [[r["host_timestamp_ns"]] + [r[c] for c in cols] for r in rd]
    a = np.array([[float(v) if v not in ("", "nan", "None") else np.nan for v in r] for r in rows])
    return a[:, 0], a[:, 1:]


def dist_m(lat1, lon1, lat2, lon2):
    k = np.pi / 180
    dx = (lon2 - lon1) * k * R_EARTH * np.cos(lat1 * k)
    dy = (lat2 - lat1) * k * R_EARTH
    return np.hypot(dx, dy)


def interp(t_src, v_src, t_dst):
    return np.interp(t_dst, t_src, v_src)


def check(tel, m):
    issues, s = [], {}
    d = {}
    for st, cols in STREAMS.items():
        t, v = load(tel / f"{m}_{st}.csv", cols)
        d[st] = (t, v)
        nan = int(np.isnan(v).sum())
        if nan:
            issues.append(f"{st}: {nan} NaN/empty")
        dt = np.diff(t) / 1e9
        s[f"{st}_hz"] = round(1 / float(np.median(dt)), 2) if len(dt) else 0
    tg, g = d["GLOBAL_POSITION_INT"]
    rel = g[:, 3] / 1e3
    fl = rel > FLIGHT_ALT_M
    tf0, tf1 = (tg[fl][0], tg[fl][-1]) if fl.any() else (0, 0)

    def in_flight(t):
        return (t >= tf0) & (t <= tf1)

    def where(t):
        # seconds after start, relative altitude, seconds after touchdown (last flight sample)
        a = float(interp(tg, rel, [t])[0])
        return f"at {(t - tg[0]) / 1e9:.0f} s, alt {a:.1f} m, {(t - tf1) / 1e9:+.0f} s to touchdown"

    # --- ranges, GLOBAL_POSITION_INT
    home_lat, home_lon = g[0, 0] / 1e7, g[0, 1] / 1e7
    home_alt = (g[0, 2] - g[0, 3]) / 1e3
    hd = dist_m(home_lat, home_lon, g[:, 0] / 1e7, g[:, 1] / 1e7)
    s["max_dist_m"] = round(float(hd.max()), 1)
    if rel.min() < -2 or rel.max() > 150:
        issues.append(f"relative_alt {rel.min():.1f}..{rel.max():.1f} m")
    alt_res = np.abs((g[:, 2] - g[:, 3]) / 1e3 - home_alt)
    if alt_res.max() > 1.0:
        issues.append(f"alt - relative_alt drifts {alt_res.max():.1f} m")
    vh = np.hypot(g[:, 4], g[:, 5]) / 100
    if vh.max() > 25 or np.abs(g[:, 6]).max() / 100 > 10:
        issues.append(f"velocity vh {vh.max():.1f} vz {np.abs(g[:, 6]).max() / 100:.1f} m/s")
    hdg = g[:, 7]
    if ((hdg < 0) | ((hdg > 35999) & (hdg != 65535))).any():
        issues.append("hdg out of range")

    # --- GPS_RAW_INT
    tp, p = d["GPS_RAW_INT"]
    pf = in_flight(tp)
    if pf.any():
        if (p[pf, 0] < 3).any():
            issues.append(f"GPS fix < 3 in flight ({int((p[pf, 0] < 3).sum())})")
        if p[pf, 8].min() < 6:
            issues.append(f"satellites {int(p[pf, 8].min())} in flight")
        if p[pf, 4].max() > 500 or p[pf, 5].max() > 1000:
            issues.append(f"eph/epv {int(p[pf, 4].max())}/{int(p[pf, 5].max())}")
    if p[:, 6].max() > 3000 or (p[:, 7] > 35999).any():
        bad = (p[:, 6] > 3000) | (p[:, 7] > 35999)
        issues.append(f"GPS vel/cog out of range: {int(bad.sum())} samples, vel {p[bad, 6].max():.0f}, "
                      f"cog {p[bad, 7].max():.0f}, first {where(tp[bad][0])}")
    # GPS vs GLOBAL_POSITION_INT (position, altitude, speed) in flight
    if pf.any() and fl.sum() > 10:
        gl = interp(tg, g[:, 0], tp[pf]) / 1e7; gn = interp(tg, g[:, 1], tp[pf]) / 1e7
        dpos = dist_m(gl, gn, p[pf, 1] / 1e7, p[pf, 2] / 1e7)
        dalt = np.abs(interp(tg, g[:, 2], tp[pf]) - p[pf, 3]) / 1e3
        dvel = np.abs(interp(tg, vh, tp[pf]) - p[pf, 6] / 100)
        s["gps_dpos_p99"] = round(float(np.percentile(dpos, 99)), 2)
        s["gps_dalt_p99"] = round(float(np.percentile(dalt, 99)), 2)
        s["gps_dvel_p99"] = round(float(np.percentile(dvel, 99)), 2)
        if s["gps_dpos_p99"] > 10 or s["gps_dalt_p99"] > 10 or s["gps_dvel_p99"] > 5:
            issues.append(f"GPS vs position: pos {s['gps_dpos_p99']} m, alt {s['gps_dalt_p99']} m, vel {s['gps_dvel_p99']} m/s (p99)")

    # --- RAW_IMU
    ti, imu = d["RAW_IMU"]
    acc = np.linalg.norm(imu[:, 0:3], axis=1)
    gyr = np.linalg.norm(imu[:, 3:6], axis=1) / 1000
    mag = np.linalg.norm(imu[:, 6:9], axis=1)
    fi = in_flight(ti)
    if acc.max() > 4000:
        issues.append(f"|acc| max {acc.max():.0f} mG {where(ti[acc.argmax()])}")
    if gyr.max() > 10:
        issues.append(f"|gyro| max {gyr.max():.1f} rad/s {where(ti[gyr.argmax()])}")
    mmed = float(np.median(mag))
    if np.abs(mag / mmed - 1).max() > 0.2:
        issues.append(f"|mag| deviates {100 * np.abs(mag / mmed - 1).max():.0f} % from median")
    s["acc_ground_mg"] = round(float(np.median(acc[:20])), 1)
    if abs(s["acc_ground_mg"] - 1000) > 50:
        issues.append(f"|acc| at rest {s['acc_ground_mg']} mG")
    if fi.any():
        s["acc_std_flight"] = round(float(acc[fi].std()), 1)
        s["gyro_p99_flight"] = round(float(np.percentile(gyr[fi], 99)), 3)
    s["mag_median"] = round(mmed, 1)

    # --- SCALED_PRESSURE: baro altitude vs relative_alt
    tb, b = d["SCALED_PRESSURE"]
    if b[:, 0].min() < 800 or b[:, 0].max() > 1050:
        issues.append(f"press_abs {b[:, 0].min():.1f}..{b[:, 0].max():.1f} hPa")
    if (b[:, 1] != 0).any():
        issues.append("press_diff != 0")
    s["press_ground"] = round(float(np.median(b[:10, 0])), 2)
    s["temp_cdeg"] = round(float(np.median(b[:, 2])), 0)
    if fl.sum() > 10:
        r = interp(tg, rel, tb)
        A = np.vstack([r, np.ones_like(r)]).T
        coef, *_ = np.linalg.lstsq(A, b[:, 0], rcond=None)
        res = b[:, 0] - A @ coef
        s["baro_slope_hpa_m"] = round(float(coef[0]), 4)
        s["baro_res_m"] = round(float(res.std() / abs(coef[0])), 2) if coef[0] else None
        if not (-0.13 < coef[0] < -0.09) or (s["baro_res_m"] is None or s["baro_res_m"] > 3):
            issues.append(f"baro vs relative_alt slope {coef[0]:.3f} hPa/m, residual {s['baro_res_m']} m")

    # --- SERVO_OUTPUT_RAW
    tsv, sv = d["SERVO_OUTPUT_RAW"]
    m14 = sv[:, :4]
    if ((m14 != 0) & ((m14 < 1000) | (m14 > 2000))).any():
        issues.append("servo1-4 outside 1000-2000")
    if (sv[:, 4:] != 0).any():
        issues.append("servo5-8 != 0")
    fs = in_flight(tsv)
    if fs.any():
        s["servo_mean_flight"] = round(float(m14[fs].mean()), 1)
        if m14[fs].min() < 1100:
            issues.append(f"servo {int(m14[fs].min())} in flight")

    # --- frozen values in flight (all feature columns of a stream identical)
    for st, (t, v) in d.items():
        f = in_flight(t)
        if f.sum() < 10:
            continue
        if st == "SCALED_PRESSURE":
            # SITL baro has no noise: constant while holding altitude (stat only)
            s["press_eq_share"] = round(float(np.mean(np.diff(v[f, 0]) == 0)), 3)
            continue
        vv, tt = v[f], t[f]
        if st == "SERVO_OUTPUT_RAW":
            vv = vv[:, :4]
        same = np.all(vv[1:] == vv[:-1], axis=1)
        run = best = 0.0
        for k, eq in enumerate(same):
            run = run + (tt[k + 1] - tt[k]) / 1e9 if eq else 0.0
            best = max(best, run)
        if best >= FROZEN_S:
            issues.append(f"{st}: frozen {best:.1f} s in flight")
    # single columns constant over the whole flight that should vary
    for st, idx in (("RAW_IMU", range(9)), ("GLOBAL_POSITION_INT", range(8))):
        t, v = d[st]; f = in_flight(t)
        if f.sum() > 10:
            const = [STREAMS[st][i] for i in idx if np.ptp(v[f, i]) == 0]
            if const:
                issues.append(f"{st}: constant in flight {const}")
    s["const_cols"] = [f"{st}.{c}" for st, (t, v) in d.items()
                       for i, c in enumerate(STREAMS[st]) if np.ptp(v[:, i]) == 0]
    s["const_vals"] = {f"{st}.{c}": float(v[0, i]) for st, (t, v) in d.items()
                       for i, c in enumerate(STREAMS[st]) if np.ptp(v[:, i]) == 0}
    s["flight_s"] = round(float((tf1 - tf0) / 1e9), 1)
    s["alt_max"] = round(float(rel.max()), 1)
    s["vh_p95"] = round(float(np.percentile(vh[fl], 95)), 2) if fl.any() else 0
    return issues, s


def main():
    repo = Path(sys.argv[1]); ds = repo / "dataset"; tel = ds / "telemetry"
    rows = [r for r in csv.DictReader(open(repo / sys.argv[2])) if r["status"] == "completed"]
    out = []
    for r in rows:
        m = r["mission_id"]
        try:
            issues, s = check(tel, m)
        except Exception as e:  # report, do not stop
            issues, s = [f"check error: {type(e).__name__}: {e}"], {}
        out.append({"id": m, "preset": r["drone_preset"], "wind": r["wind_profile"],
                    "profile": r["profile"], "issues": issues, **s})
    json.dump(out, open(sys.argv[3], "w"))
    bad = [o for o in out if o["issues"]]
    print(f"{len(out)} missions, {len(bad)} with issues")
    for o in bad:
        print(o["id"], o["preset"], o["wind"], "|", "; ".join(o["issues"]))


if __name__ == "__main__":
    main()
