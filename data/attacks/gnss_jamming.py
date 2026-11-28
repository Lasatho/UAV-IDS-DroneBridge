#!/usr/bin/env python3
"""
T2 GNSS Jamming — Disable GPS receiver in SITL.

Simulates RF-level GNSS jamming by setting SIM_GPS1_JAM=1 via
MAVLink PARAM_SET. On termination, clears jamming (SIM_GPS1_JAM=0).

Expected drone behavior: GPS fix lost, EKF degrades to dead-reckoning,
position drift, potential mode change to ALT_HOLD or LAND.

Requires a separate MAVLink connection to SITL. The orchestrator passes
the direct SITL port (tcp:127.0.0.1:5762), so that the parameter commands
do not travel over the DroneBridge link.

Usage (standalone test):
    python3 gnss_jamming.py --connection udpin:127.0.0.1:14554

Normally started/stopped by the orchestrator.
"""

import argparse
import signal
import time

from pymavlink import mavutil

keep_running = True


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
                print(f"[GNSS_JAM] {param_id} = {ack.param_value}",
                      flush=True)
                return True
    print(f"[GNSS_JAM] Failed to set {param_id} = {value}", flush=True)
    return False


def run_gnss_jamming(connection: str):
    conn = mavutil.mavlink_connection(connection)
    conn.wait_heartbeat()
    print(f"[GNSS_JAM] Connected to system {conn.target_system}", flush=True)

    # Jam GPS
    set_param(conn, "SIM_GPS1_JAM", 1)
    print("[GNSS_JAM] GPS JAMMED", flush=True)

    # Hold until terminated
    while keep_running:
        time.sleep(1)

    # Clear jamming on exit
    print("[GNSS_JAM] Restoring GPS...", flush=True)
    set_param(conn, "SIM_GPS1_JAM", 0, retries=5)
    print("[GNSS_JAM] GPS RESTORED", flush=True)
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="T2 GNSS Jamming")
    parser.add_argument("--connection", default="udpin:127.0.0.1:14554")
    parser.add_argument("--mission-id", default="manual")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    run_gnss_jamming(args.connection)


if __name__ == "__main__":
    main()