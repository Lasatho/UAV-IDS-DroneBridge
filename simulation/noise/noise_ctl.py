"""Noise model of the EKF3 re-flight (Lasse 2026-11-26, Vault sensorrauschen) on one SITL instance via its direct
SITL port (5762 + 10 k). usage: noise_ctl.py PORT set | get | verify SECONDS
set: writes PARAMS (saved to EEPROM, persists over the orchestrator's reboots), verify: params + spread at rest."""
import math, statistics, sys, time
from pymavlink import mavutil

PARAMS = {
    "SCR_ENABLE": 1,              # gps_noise.lua (GPS Gauss-Markov 0.5 m, tau 30 s)
    "SIM_VIB_MOT_MASK": 15,       # IMU noise of SITL acts with Gazebo/JSON only with the mask
    "MOT_BAT_VOLT_MAX": 0,        # mask enables the sim battery -> no voltage compensation (roll oscillation)
    "SIM_GYR1_RND": 5,
    "SIM_BARO_RND": 0.22,
    "SIM_GPS1_VERR_X": 0.05,
    "SIM_GPS1_VERR_Y": 0.05,
    "SIM_GPS1_VERR_Z": 0.07,
    "ATC_RAT_RLL_P": 0.064125,    # roll gains 0.475 x default (Befund 49, reserve 2)
    "ATC_RAT_RLL_I": 0.064125,
    "ATC_RAT_RLL_D": 0.00171,
}
CHECK = list(PARAMS) + ["AHRS_EKF_TYPE"]

port, cmd = sys.argv[1], sys.argv[2]
c = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", source_system=251)
c.wait_heartbeat(timeout=60)


def get(n):
    for _ in range(3):
        c.mav.param_request_read_send(c.target_system, c.target_component, n.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < 3:
            m = c.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if m and m.param_id == n:
                return m.param_value
    return None


def show():
    bad = 0
    for n in CHECK:
        v = get(n); want = PARAMS.get(n, 3)
        ok = v is not None and abs(v - want) <= 1e-4 * max(1, abs(want))
        bad += not ok
        print(f"  {n:18s} {v} {'ok' if ok else 'WANT ' + str(want)}")
    return bad


if cmd == "set":
    for n, v in PARAMS.items():
        for _ in range(5):
            c.mav.param_set_send(c.target_system, c.target_component, n.encode(), float(v),
                                 mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            time.sleep(0.5)
            g = get(n)
            if g is not None and abs(g - v) <= 1e-4 * max(1, abs(v)):
                break
    time.sleep(1)
    print("PARAMS_BAD", show())
elif cmd == "get":
    print("PARAMS_BAD", show())
elif cmd == "verify":
    bad = show()
    for mid in (24, 27):  # GPS_RAW_INT, RAW_IMU at 5 Hz
        c.mav.command_long_send(c.target_system, c.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mid, 200000, 0, 0, 0, 0, 0)
    lat, lon, gx, az = [], [], [], []
    t0 = time.time()
    while time.time() - t0 < float(sys.argv[3]):
        m = c.recv_match(type=["GPS_RAW_INT", "RAW_IMU", "STATUSTEXT"], blocking=True, timeout=1)
        if not m:
            continue
        t = m.get_type()
        if t == "GPS_RAW_INT" and m.fix_type >= 3:
            lat.append(m.lat * 1e-7); lon.append(m.lon * 1e-7)
        elif t == "RAW_IMU":
            gx.append(m.xgyro * 1e-3); az.append(m.zacc * 9.80665e-3)
        elif t == "STATUSTEXT":
            print("  STATUSTEXT", m.text)
    sd = lambda v: statistics.pstdev(v) if len(v) > 2 else float("nan")
    gn = sd(lat) * 111320; ge = sd(lon) * 111320 * math.cos(math.radians(statistics.mean(lat) if lat else 0))
    print(f"  rest {sys.argv[3]} s: GPS n={len(lat)} std N {gn:.3f} E {ge:.3f} m | RAW_IMU n={len(gx)} "
          f"xgyro std {sd(gx):.4f} rad/s zacc std {sd(az):.4f} m/s2")
    ok = bad == 0 and 0.05 < gn < 2 and 0.05 < ge < 2 and sd(gx) > 0.001 and sd(az) > 0.003
    print("VERIFY", "OK" if ok else "FAILED")
