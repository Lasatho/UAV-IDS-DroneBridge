#!/usr/bin/env python3
"""
T1 GNSS Spoofing — Gradually shift simulated GPS position.

Simulates RF-level GNSS spoofing by incrementally modifying
SIM_GPS1_GLTCH_X (lat) and SIM_GPS1_GLTCH_Y (lon) parameters.
Drift is applied gradually to avoid EKF innovation check rejection.

On termination, resets glitch parameters to 0.

Expected drone behavior: drone follows spoofed coordinates, physical
sensors (IMU, barometer) contradict reported position — the CPS
inconsistency signal the world-model IDS should detect.

Requires a separate MAVLink connection to SITL. The orchestrator passes
the direct SITL port (tcp:127.0.0.1:5762), so that the parameter commands
do not travel over the DroneBridge link.

Usage (standalone test):
    python3 gnss_spoofing.py --connection udpin:127.0.0.1:14554

Normally started/stopped by the orchestrator.
"""

import argparse
import os
import signal
import time
import math
import random

from pymavlink import mavutil

keep_running = True

# Spoofing parameters
DRIFT_RATE = 0.00002      # degrees per step (~2.2m per step at equator)
DRIFT_INTERVAL = 2.0      # seconds between drift steps
MAX_DRIFT = 0.001         # max drift in degrees (~111m)


def handle_signal(sig, frame):
    global keep_running
    keep_running = False


def set_param(conn, param_id: str, value: float, retries=3, timeout=1.0):
    """Set a SITL parameter and wait until SITL reports the new value.

    Only a PARAM_VALUE with this name and value counts: older PARAM_VALUEs
    still queued would end the wait at once. SITL drops received but not yet
    parsed bytes when the TCP connection closes, so the reset must be
    confirmed before close() (Befund 33).
    """
    while conn.recv_match(blocking=False) is not None:
        pass
    for attempt in range(retries):
        conn.mav.param_set_send(
            conn.target_system,
            conn.target_component,
            param_id.encode("utf-8"),
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            ack = conn.recv_match(type="PARAM_VALUE", blocking=True,
                                  timeout=deadline - time.time())
            if (ack and ack.param_id.strip("\x00") == param_id
                    and abs(ack.param_value - value) <= 1e-6 * abs(value)):
                return True
    print(f"[GNSS_SPOOF] Failed to set {param_id} = {value}", flush=True)
    return False


def run_gnss_spoofing(connection: str):
    conn = mavutil.mavlink_connection(connection)
    conn.wait_heartbeat()
    print(f"[GNSS_SPOOF] Connected to system {conn.target_system}", flush=True)

    # Random drift direction
    drift_angle = random.uniform(0, 2 * math.pi)
    drift_dx = DRIFT_RATE * math.cos(drift_angle)
    drift_dy = DRIFT_RATE * math.sin(drift_angle)

    current_x = 0.0
    current_y = 0.0
    step = 0

    print(f"[GNSS_SPOOF] Drift direction: {math.degrees(drift_angle):.0f}°, "
          f"rate: {DRIFT_RATE}°/step", flush=True)

    while keep_running:
        # Apply incremental drift
        current_x += drift_dx
        current_y += drift_dy

        # Clamp to max drift
        current_x = max(-MAX_DRIFT, min(MAX_DRIFT, current_x))
        current_y = max(-MAX_DRIFT, min(MAX_DRIFT, current_y))

        # One attempt per step (the next step sends the new value anyway),
        # so that step and reset stay within the orchestrator's 15 s
        # SIGTERM timeout.
        set_param(conn, "SIM_GPS1_GLTCH_X", current_x, retries=1)
        set_param(conn, "SIM_GPS1_GLTCH_Y", current_y, retries=1)

        step += 1
        drift_m = math.sqrt(current_x**2 + current_y**2) * 111000
        if step % 5 == 0:
            print(f"[GNSS_SPOOF] Step {step}: glitch=({current_x:.6f}, "
                  f"{current_y:.6f}), drift={drift_m:.1f}m", flush=True)

        time.sleep(DRIFT_INTERVAL)

    # Reset on exit
    print("[GNSS_SPOOF] Resetting GPS glitch...", flush=True)
    ok_x = set_param(conn, "SIM_GPS1_GLTCH_X", 0, retries=5)
    ok_y = set_param(conn, "SIM_GPS1_GLTCH_Y", 0, retries=5)
    print(f"[GNSS_SPOOF] GPS glitch RESET (confirmed: {ok_x and ok_y})",
          flush=True)
    conn.close()


LUA_CMD = "spooftrig/cmd"  # shared with the SITL container (simulation/ardupilot/scripts/gnss_spoof.lua)


def run_lua(variant: str):
    """Drift-rate study (2026-11-22): the drift is applied by a Lua script inside SITL, so no parameter traffic
    appears on the MAVLink channels (Vault eval_gnss_spoofing, finding 46). Writes the command file, waits for
    SIGTERM, removes it (the script resets the glitch)."""
    import os
    kv = dict(x.split("=") for x in variant[4:].split(":"))
    rate, target = float(kv["rate"]), float(kv["target"])
    angle = random.uniform(0, 2 * math.pi)
    tmp = LUA_CMD + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{rate} {angle} {target}\n")
    os.replace(tmp, LUA_CMD)
    print(f"[GNSS_SPOOF] lua mode: rate {rate} m/s, direction {math.degrees(angle):.0f} deg, target {target} m", flush=True)
    try:
        while keep_running:
            time.sleep(0.2)
    finally:
        try:
            os.remove(LUA_CMD)
        except FileNotFoundError:
            pass
        print("[GNSS_SPOOF] lua mode: command removed (glitch reset by the script)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="T1 GNSS Spoofing")
    parser.add_argument("--connection", default="udpin:127.0.0.1:14554")
    parser.add_argument("--mission-id", default="manual")
    parser.add_argument("--variant", default=None,
                        help="lua:rate=<m/s>:target=<m> -> drift set by the SITL Lua script (no MAVLink)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if args.variant and args.variant.startswith("lua:"):
        run_lua(args.variant)
    elif os.environ.get("GNSS_SPOOF_LUA") == "1":
        run_lua("lua:rate=1.1:target=1000")
    else:
        run_gnss_spoofing(args.connection)


if __name__ == "__main__":
    main()