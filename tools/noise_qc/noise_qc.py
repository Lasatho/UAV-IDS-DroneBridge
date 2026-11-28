"""Plausibility check of the noisy sensor values in the v4 flights (Lasse 2026-11-26, Vault sensorrauschen
"Prüfmethodik"). Incremental: every mission with status completed in the given manifests and a phase label is checked
once (cache OUT/missions.jsonl, key mission + label mtime); then OUT/summary.json and OUT/summary.md are rewritten.

Per mission (host time of the telemetry CSVs, GCS view):
  R  rest = before the first spin-up (servo1 > 1100), at least 5 s: spread of GPS north/east/alt, barometer height,
     gyro and accelerometer per axis. ALERT if the noise is missing (container recreated without noise_apply.sh),
     FLAG if outside the plausible band.
  S  spikes before the attack (attack start - 5 s to end of mission excluded, its effects belong to the attack; label times
     mapped from epoch onto the monotonic telemetry clock by the mission start, fixed 2026-11-26): GPS innovation e = dp_GPS - v_EKF dt (robust sigma via
     MAD, max |e|, excess kurtosis), barometer step against EKF vz, EKF position step against EKF velocity, IMU extremes.
  F  flight: roll limit cycle per 30 s block (Befund 49 criterion: narrow-line share > 0.5 and xgyro std > 3 x ygyro
     std), GPS altitude minus barometer height spread.
  P  pairing with the Typ-10 original (same mission ID, REF phase label): duration ratio, success, timeout.
Aggregate: per instance medians and the ratio of the measured GPS rest variance to the Gauss-Markov expectation for
the rest length (should be ~1 over many missions), forced-disarm share per wind profile (normal flights) against Typ 10.
usage: noise_qc.py --data DIR --manifests GLOB [--ref DIR] --out DIR [--sigma 0.5 --tau 30] [--procs 4] [--live]\n--live: missions in flight only (printed, not cached) for an early check."""
import argparse, csv, glob, json, math, os, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--manifests", required=True)
ap.add_argument("--ref", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--tau", type=float, default=30.0)
ap.add_argument("--procs", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--live", action="store_true", help="only the missions in flight (telemetry, no label yet); not cached")
A = ap.parse_args()

# Plausible bands (Vault sensorrauschen: PX4 logs u-blox M8/M9/M10 p10-p90, UAV-FD, configured model)
MISSING = {"gps_n": 0.02, "gps_e": 0.02, "gps_alt": 0.05, "baro": 0.03, "gyr_x": 0.001, "acc_z": 0.003}
BAND = {"gps_n": (0.0, 2.0), "gps_e": (0.0, 2.0), "gps_alt": (0.05, 4.0), "baro": (0.03, 0.5),
        "gyr_x": (0.001, 0.02), "gyr_y": (0.001, 0.02), "gyr_z": (0.001, 0.02),
        "acc_x": (0.003, 0.08), "acc_y": (0.003, 0.08), "acc_z": (0.003, 0.08)}
# 2 x p99 over 137 v4 missions (TR, 2026-11-28, GPS time): p99 innovation 1.11 m, kurtosis 3.79, baro step 0.61 m,
# EKF step 0.82 m (before: 3.0 / 12 / 2.0 / 2.0 set without data; Vault sensorrauschen "GPS-Kurtosis")
GPS_JUMP_M, GPS_KURT, BARO_STEP_M, EKF_JUMP_M = 2.2, 7.5, 1.2, 1.6
GYR_MAX, ACC_MAX = 4.0, 33.0          # rad/s, m/s^2 in flight outside attacks; min(old 4/40, 2 x p99 flight 5.3/33)
DUR_RATIO = (0.8, 1.5)                 # new / Typ 10 (EKF3 alone: +17 s on ~160 s missions)


def load(path, cols):
    try:
        with open(path) as f:
            r = csv.reader(f); h = next(r); idx = [h.index(c) for c in cols]
            a = np.array([[float(row[i]) for i in idx] for row in r if len(row) == len(h)])
        return a if a.ndim == 2 and len(a) else None
    except Exception:
        return None


def fnum(x):
    try: return float(x)
    except Exception: return None


def mask_out(t, wins):
    m = np.ones(len(t), bool)
    for a, b in wins: m &= ~((t >= a) & (t <= b))
    return m


def kurt(x):
    x = x - x.mean(); v = (x ** 2).mean()
    return float((x ** 4).mean() / v ** 2 - 3) if v > 0 else 0.0


def check(job):
    mid, d, ref = job
    tel = f"{d}/telemetry/{mid}_"
    lp = f"{d}/phase_labels/{mid}.json"
    L = json.load(open(lp)) if os.path.exists(lp) and ref != "__live__" else {"attack_type": "live"}
    res = {"mission": mid, "attack": L.get("attack_type"), "instance": L.get("instance", "0"), "flags": [], "alerts": []}
    g = load(tel + "GPS_RAW_INT.csv", ["host_timestamp_ns", "fix_type", "lat", "lon", "alt", "time_usec"])
    p = load(tel + "GLOBAL_POSITION_INT.csv", ["host_timestamp_ns", "lat", "lon", "relative_alt", "vx", "vy", "vz"])
    i = load(tel + "RAW_IMU.csv", ["host_timestamp_ns", "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"])
    b = load(tel + "SCALED_PRESSURE.csv", ["host_timestamp_ns", "press_abs"])
    s = load(tel + "SERVO_OUTPUT_RAW.csv", ["host_timestamp_ns", "servo1_raw"])
    if any(x is None or len(x) < 50 for x in (g, p, i, b, s)):
        res["alerts"].append("telemetry_missing"); return res
    for x in (g, p, i, b, s): x[:, 0] *= 1e-9
    t0_tel = min(x[0, 0] for x in (g, p, i, b, s))
    lat0 = g[g[:, 1] >= 3][0, 2] * 1e-7 if (g[:, 1] >= 3).any() else -35.36
    kx, ky = 111320.0, 111320.0 * math.cos(math.radians(lat0))
    gn, ge, ga = g[:, 2] * 1e-7 * kx, g[:, 3] * 1e-7 * ky, g[:, 4] * 1e-3
    bh = 44330.0 * (1 - (b[:, 1] / 1013.25) ** 0.1903)
    acc, gyr = i[:, 1:4] * 9.80665e-3, i[:, 4:7] * 1e-3
    # R: rest phase
    spin = s[s[:, 1] > 1100, 0]
    t_spin = spin[0] if len(spin) else s[-1, 0]
    t0 = max(g[0, 0], i[0, 0], b[0, 0]); t1 = t_spin - 1.0
    res["rest_s"] = round(t1 - t0, 1)
    if t1 - t0 >= 5:
        rg = (g[:, 0] >= t0) & (g[:, 0] < t1) & (g[:, 1] >= 3)
        ri = (i[:, 0] >= t0) & (i[:, 0] < t1); rb = (b[:, 0] >= t0) & (b[:, 0] < t1)
        if rg.sum() > 5 and ri.sum() > 5 and rb.sum() > 5:
            r = {"gps_n": gn[rg].std(), "gps_e": ge[rg].std(), "gps_alt": ga[rg].std(), "baro": bh[rb].std()}
            for k, a in zip("xyz", range(3)):
                r[f"gyr_{k}"] = gyr[ri, a].std(); r[f"acc_{k}"] = acc[ri, a].std()
            res["rest"] = {k: round(float(v), 4) for k, v in r.items()}
            for k, lim in MISSING.items():
                if r[k] < lim: res["alerts"].append(f"noise_missing:{k}={r[k]:.4f}")
            for k, (lo, hi) in BAND.items():
                if not lo <= r[k] <= hi and not any(a.startswith(f"noise_missing:{k}") for a in res["alerts"]):
                    res["flags"].append(f"rest_band:{k}={r[k]:.4f}")
    else:
        res["flags"].append("rest_short")
    # attack windows to exclude from spike checks
    # attack effects (EKF resets after spoofing/jamming, failsafe landings) count as attack: exclude attack start
    # to the end of the mission from the spike checks
    # labels are in epoch seconds, the telemetry host_timestamp_ns on the monotonic clock of the host: map label times
    # onto the telemetry clock by the mission start (first sample ~ start_time, error ~1 s; 5 s margin before the attack)
    off = (fnum(L.get("start_time")) or 0.0) - t0_tel
    wins = []
    a0 = fnum(L.get("attack_start"))
    if a0: wins.append((a0 - off - 5.0, 1e12))
    # S: GPS innovation against EKF velocity
    # dt from the GPS sample time (time_usec), not from the receive time: the 5 Hz GPS reaches the GCS in a 4 Hz stream,
    # so two received rows lie 0.2 or 0.4 s apart in GPS time at 0.25 s receive time; with the receive time the
    # innovation grew with the speed (|e| ~ 2 m at 12 m/s) and gave excess kurtosis 12-43 (fixed 2026-11-28, Vault
    # sensorrauschen "GPS-Kurtosis"). Rows without a new GPS time are dropped.
    m = (g[:, 1] >= 3) & mask_out(g[:, 0], wins)
    m &= np.r_[True, np.diff(g[:, 5]) > 0]
    tg = g[m, 0]; tu = g[m, 5] * 1e-6
    if len(tg) > 20:
        vx = np.interp(tg, p[:, 0], p[:, 4] * 0.01); vy = np.interp(tg, p[:, 0], p[:, 5] * 0.01)
        dt = np.diff(tu); ok = (dt > 0.05) & (dt < 2.0)
        e = np.hypot(np.diff(gn[m]) - vx[1:] * dt, np.diff(ge[m]) - vy[1:] * dt)[ok]
        if len(e) > 10:
            res["gps_innov_max"] = round(float(e.max()), 3); res["gps_innov_mad"] = round(float(np.median(np.abs(e - np.median(e)))), 4)
            de = np.diff(gn[m])[ok] - (vx[1:] * dt)[ok]; res["gps_innov_kurt"] = round(kurt(de), 2)
            if e.max() > GPS_JUMP_M: res["flags"].append(f"gps_jump:{e.max():.2f}m")
            if res["gps_innov_kurt"] > GPS_KURT: res["flags"].append(f"gps_kurtosis:{res['gps_innov_kurt']}")
    # barometer step against EKF vz
    mb = mask_out(b[:, 0], wins); tb = b[mb, 0]
    if len(tb) > 20:
        vz = np.interp(tb, p[:, 0], p[:, 6] * 0.01)  # NED down positive
        dt = np.diff(tb); ok = (dt > 0.05) & (dt < 2.0)
        st = np.abs(np.diff(bh[mb]) + vz[1:] * dt)[ok]
        if len(st): res["baro_step_max"] = round(float(st.max()), 3)
        if len(st) and st.max() > BARO_STEP_M: res["flags"].append(f"baro_step:{st.max():.2f}m")
    # EKF position step against EKF velocity
    mp = mask_out(p[:, 0], wins); tp = p[mp, 0]
    if len(tp) > 20:
        pn, pe = p[mp, 1] * 1e-7 * kx, p[mp, 2] * 1e-7 * ky; dt = np.diff(tp); ok = (dt > 0.05) & (dt < 2.0)
        js = np.hypot(np.diff(pn) - p[mp, 4][1:] * 0.01 * dt, np.diff(pe) - p[mp, 5][1:] * 0.01 * dt)[ok]
        if len(js): res["ekf_step_max"] = round(float(js.max()), 3)
        if len(js) and js.max() > EKF_JUMP_M: res["flags"].append(f"ekf_jump:{js.max():.2f}m")
    # IMU extremes outside attack windows and before disarm
    t_dis = (fnum(L.get("disarm_time")) - off) if fnum(L.get("disarm_time")) else i[-1, 0]
    # flight = relative altitude > 0.5 m; the maxima before 2026-11-28 were all ground contact after the touchdown
    # (alt ~0.1 m, motors running in wind until the forced disarm, Vault sensorrauschen "GPS-Kurtosis"): info only
    mi = mask_out(i[:, 0], wins) & (i[:, 0] < t_dis)
    air = np.interp(i[:, 0], p[:, 0], p[:, 3] * 1e-3) > 0.5
    for sfx, mm in (("", mi & air), ("_ground", mi & ~air)):
        if mm.sum():
            gm, am = float(np.abs(gyr[mm]).max()), float(np.linalg.norm(acc[mm], axis=1).max())
            res["gyr_max" + sfx], res["acc_max" + sfx] = round(gm, 3), round(am, 2)
            if not sfx and gm > GYR_MAX: res["flags"].append(f"gyr_max:{gm:.2f}")
            if not sfx and am > ACC_MAX: res["flags"].append(f"acc_max:{am:.1f}")
    # F: roll limit cycle per 30 s flight block, GPS alt - baro height in flight
    fl = s[s[:, 1] > 1100, 0]
    lc = nb = 0
    if len(fl) > 10:
        ta, tb_ = fl[0] + 30, fl[-1] - 10
        t = ta
        while t + 30 <= tb_:
            k = (i[:, 0] >= t) & (i[:, 0] < t + 30)
            if k.sum() >= 60:
                x = gyr[k, 0] - gyr[k, 0].mean(); y = gyr[k, 1]
                fs = k.sum() / 30.0; P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
                band = (f >= 0.3) & (f <= 2.0); share = 0.0
                if band.any() and P[1:].sum() > 0:
                    pk = f[band][np.argmax(P[band])]; share = P[(f >= pk - 0.1) & (f <= pk + 0.1)].sum() / P[1:].sum()
                nb += 1; lc += int(share > 0.5 and x.std() > 3 * y.std())
            t += 30
        mg = (g[:, 0] >= fl[0]) & (g[:, 0] <= fl[-1]) & (g[:, 1] >= 3)
        if mg.sum() > 20:
            dh = ga[mg] - np.interp(g[mg, 0], b[:, 0], bh); res["alt_gps_baro_std"] = round(float(dh.std()), 3)
            if dh.std() > 3.0: res["flags"].append(f"alt_gps_baro:{dh.std():.2f}m")
    res["lc_blocks"], res["blocks"] = lc, nb
    if lc: res["flags"].append(f"roll_limit_cycle:{lc}/{nb}")
    # P: pairing with Typ 10
    res["duration_s"] = round(fnum(L.get("duration_s")) or 0, 1)
    res["success"] = str(L.get("mission_success")) == "True"; res["timed_out"] = str(L.get("timed_out")) == "True"
    res["forced_disarm"] = str(L.get("landed_by_orchestrator")) == "True"
    res["wind"] = (L.get("wind") or "")
    if L.get("attack_type") == "live": return res
    if res["timed_out"]: res["flags"].append("timed_out")
    if not res["success"] and res["attack"] == "none": res["flags"].append("normal_not_successful")
    if ref and ref != "__live__" and os.path.exists(f"{ref}/phase_labels/{mid}.json"):
        R0 = json.load(open(f"{ref}/phase_labels/{mid}.json")); d0 = fnum(R0.get("duration_s"))
        if d0:
            res["dur_ratio"] = round(res["duration_s"] / d0, 3)
            if not DUR_RATIO[0] <= res["dur_ratio"] <= DUR_RATIO[1]: res["flags"].append(f"duration_ratio:{res['dur_ratio']}")
        res["ref_forced_disarm"] = str(R0.get("landed_by_orchestrator")) == "True"
    return res


def gm_expect(T, sigma, tau):
    """Expected sample variance of a stationary Gauss-Markov process over a window of length T."""
    x = T / tau
    return sigma ** 2 * (1 - 2 / x + 2 / x ** 2 * (1 - math.exp(-x)))


def main():
    os.makedirs(A.out, exist_ok=True)
    cache_p = f"{A.out}/missions.jsonl"; cache = {}
    if os.path.exists(cache_p):
        for l in open(cache_p):
            try: r = json.loads(l); cache[r["mission"]] = r
            except Exception: pass
    done = []
    for f in sorted(glob.glob(A.manifests)):
        if ".bak" in f: continue
        for r in csv.DictReader(open(f)):
            if r.get("status") == "completed": done.append(r["mission_id"])
    meta = {}
    for f in sorted(glob.glob(A.manifests)):
        if ".bak" in f: continue
        for r in csv.DictReader(open(f)): meta.setdefault(r["mission_id"], r)
    if A.live:
        new = glob.glob(f"{A.data}/telemetry/*_RAW_IMU.csv"); now = time.time()
        ids = [os.path.basename(f)[:-len("_RAW_IMU.csv")] for f in new if now - os.path.getmtime(f) < 900]
        with Pool(A.procs) as pool:
            rs = pool.map(check, [(m, A.data, "__live__") for m in ids])
        for r in rs:
            print(r["mission"], "ALERT" if r["alerts"] else ("FLAG" if r["flags"] else "OK"), r.get("rest"), r["alerts"] + r["flags"],
                  {k: r.get(k) for k in ("rest_s", "gps_innov_max", "gps_innov_mad", "gps_innov_kurt", "baro_step_max", "ekf_step_max",
                                         "gyr_max", "acc_max", "alt_gps_baro_std", "lc_blocks", "blocks")})
        return
    jobs = []
    for mid in done:
        lp = f"{A.data}/phase_labels/{mid}.json"
        if not os.path.exists(lp): continue
        mt = round(os.path.getmtime(lp))
        if mid in cache and cache[mid].get("_mtime") == mt: continue
        jobs.append((mid, A.data, A.ref))
    if A.limit: jobs = jobs[:A.limit]
    t0 = time.time()
    with Pool(A.procs) as pool:
        new = pool.map(check, jobs, chunksize=1) if jobs else []
    for r in new:
        r["_mtime"] = round(os.path.getmtime(f"{A.data}/phase_labels/{r['mission']}.json"))
        r["wind_profile"] = meta.get(r["mission"], {}).get("wind_profile", "?"); cache[r["mission"]] = r
    with open(cache_p, "w") as o:
        for r in cache.values(): o.write(json.dumps(r) + "\n")
    rs = [cache[m] for m in done if m in cache]
    summarize(rs, len(new), time.time() - t0)


def summarize(rs, n_new, secs):
    status = lambda r: "ALERT" if r["alerts"] else ("FLAG" if r["flags"] else "OK")
    by_inst = defaultdict(list)
    for r in rs: by_inst[str(r.get("instance", "0"))].append(r)
    S = {"time": time.strftime("%Y-%m-%d %H:%M"), "checked": len(rs), "new": n_new, "secs": round(secs, 1),
         "status": dict(Counter(status(r) for r in rs)), "instances": {}, "flag_types": {}, "alerts": [], "flagged": []}
    ft = Counter()
    for r in rs:
        for f in r["flags"] + r["alerts"]: ft[f.split(":")[0]] += 1
    S["flag_types"] = dict(ft.most_common())
    keys = ["gps_n", "gps_e", "baro", "gyr_x", "acc_z"]
    for k, L in sorted(by_inst.items()):
        rest = [r["rest"] for r in L if "rest" in r]
        med = {q: round(float(np.median([x[q] for x in rest])), 4) for q in keys} if rest else {}
        # measured GPS rest variance / Gauss-Markov expectation for the rest length (pooled)
        num = sum(r["rest"]["gps_n"] ** 2 + r["rest"]["gps_e"] ** 2 for r in L if "rest" in r)
        den = sum(2 * gm_expect(r["rest_s"], A.sigma, A.tau) for r in L if "rest" in r and r["rest_s"] > 0)
        S["instances"][k] = {"n": len(L), "status": dict(Counter(status(r) for r in L)), "rest_median": med,
                             "gps_var_ratio": round(num / den, 2) if den else None,
                             "dur_ratio_median": round(float(np.median([r["dur_ratio"] for r in L if "dur_ratio" in r])), 3)
                             if any("dur_ratio" in r for r in L) else None,
                             "roll_lc_blocks": f"{sum(r['lc_blocks'] for r in L)}/{sum(r['blocks'] for r in L)}"}
    fd = defaultdict(lambda: [0, 0, 0, 0])
    for r in rs:
        if r["attack"] != "none": continue
        w = fd[r.get("wind_profile", "?")]; w[0] += 1; w[1] += r["forced_disarm"]
        if "ref_forced_disarm" in r: w[2] += 1; w[3] += r["ref_forced_disarm"]
    S["forced_disarm_normal"] = {w: {"n": v[0], "new_pct": round(100 * v[1] / v[0], 1),
                                     "type10_pct_same_ids": round(100 * v[3] / v[2], 1) if v[2] else None} for w, v in fd.items()}
    S["alerts"] = [(r["mission"], r["instance"], r["alerts"]) for r in rs if r["alerts"]][:30]
    S["flagged"] = [(r["mission"], r["instance"], r["attack"], r["flags"]) for r in rs if r["flags"] and not r["alerts"]][:30]
    json.dump(S, open(f"{A.out}/summary.json", "w"), indent=1)
    md = [f"# Rauschprüfung {S['time']}", f"Geprüft {S['checked']} Missionen (neu {n_new}, {S['secs']} s). Status: {S['status']}",
          "", "| Instanz | n | Status | GPS N/O Ruhe [m] | Baro [m] | Drehrate x | Beschl. z | GPS-Varianz / Erwartung | Dauer / Typ 10 | Grenzzyklus |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in S["instances"].items():
        m = v["rest_median"]
        md.append(f"| {k} | {v['n']} | {v['status']} | {m.get('gps_n')}/{m.get('gps_e')} | {m.get('baro')} | {m.get('gyr_x')} | "
                  f"{m.get('acc_z')} | {v['gps_var_ratio']} | {v['dur_ratio_median']} | {v['roll_lc_blocks']} |")
    md += ["", "Zwangsabschaltung Normalflüge je Wind (neu / Typ 10 gleiche IDs, Typ-10-Gesamt: calm 0, light 43, moderate 100, gusty 98 %): "
           + "; ".join(f"{w} {v['new_pct']}/{v['type10_pct_same_ids']} % (n {v['n']})" for w, v in sorted(S["forced_disarm_normal"].items())),
           "", f"Befundarten: {S['flag_types'] or 'keine'}"]
    if S["alerts"]: md += ["", "**ALERT:**"] + [f"- {m} (Instanz {k}): {', '.join(a)}" for m, k, a in S["alerts"]]
    if S["flagged"]: md += ["", "FLAG (max. 30):"] + [f"- {m} (Instanz {k}, {a}): {', '.join(f)}" for m, k, a, f in S["flagged"]]
    open(f"{A.out}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
