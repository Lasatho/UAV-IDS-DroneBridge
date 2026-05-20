#!/usr/bin/env python3
"""
telemetry_logger.py — pymavlink-based sensor data logger for dataset generation.

Listens on UDP:14552 (forwarded by MAVProxy) and writes RAW_IMU, GPS_RAW_INT,
SCALED_PRESSURE, and SERVO_OUTPUT_RAW messages to CSV files.

Usage:
    python3 telemetry_logger.py --output /tmp/telemetry --run-id run_001
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path
from pymavlink import mavutil
import logging

error_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Messages to log. Extend this list if additional raw sensor data is needed.
LOGGED_MESSAGES = {
    "RAW_IMU": [
        "time_usec", "xacc", "yacc", "zacc",
        "xgyro", "ygyro", "zgyro",
        "xmag", "ymag", "zmag"
    ],
    "GPS_RAW_INT": [
        "time_usec", "fix_type", "lat", "lon", "alt",
        "eph", "epv", "vel", "cog", "satellites_visible"
    ],
    "SCALED_PRESSURE": [
        "time_boot_ms", "press_abs", "press_diff", "temperature"
    ],
    "SERVO_OUTPUT_RAW": [
        "time_usec",
        "servo1_raw", "servo2_raw", "servo3_raw", "servo4_raw",
        "servo5_raw", "servo6_raw", "servo7_raw", "servo8_raw"
    ],
    "GLOBAL_POSITION_INT": [
    "time_boot_ms", "lat", "lon", "alt", "relative_alt",
    "vx", "vy", "vz", "hdg"
    ],
}


class TelemetryLogger:
    def __init__(self, output_dir: Path, run_id: str, connection_string: str):
        self.output_dir = output_dir
        self.run_id = run_id
        self.connection_string = connection_string
        self.writers = {}
        self.files = {}
        self.keep_running = True
        self.message_counts = {msg: 0 for msg in LOGGED_MESSAGES}

    def setup_files(self):
        """Create one CSV file per message type with header row."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for msg_name, fields in LOGGED_MESSAGES.items():
            file_path = self.output_dir / f"{self.run_id}_{msg_name}.csv"
            f = open(file_path, "w", newline="")
            # host_timestamp added as common sync anchor across all CSVs
            writer = csv.writer(f)
            writer.writerow(["host_timestamp_ns"] + fields)
            self.files[msg_name] = f
            self.writers[msg_name] = writer
        error_logger.info(f"[+] Writing to {self.output_dir}/")

    def close_files(self):
        for f in self.files.values():
            f.close()

    def connect(self):
        """Establish MAVLink connection via UDP."""
        print(f"[*] Connecting to {self.connection_string}...")
        self.conn = mavutil.mavlink_connection(self.connection_string)
        # Wait for first heartbeat to confirm link is alive
        self.conn.wait_heartbeat()
        print(f"[+] Heartbeat received from system "
              f"{self.conn.target_system}:{self.conn.target_component}")
        error_logger.info(f"[+] Heartbeat received from system "
                         f"{self.conn.target_system}:{self.conn.target_component}")

    def log_message(self, msg):
        """Write a single MAVLink message to its corresponding CSV."""
        msg_type = msg.get_type()
        if msg_type not in LOGGED_MESSAGES:
            return

        fields = LOGGED_MESSAGES[msg_type]
        # Monotonic host timestamp — single clock source across all messages
        # avoids MAVLink message-internal timestamp inconsistencies
        host_ts = time.monotonic_ns()
        row = [host_ts] + [getattr(msg, f, None) for f in fields]
        self.writers[msg_type].writerow(row)
        self.message_counts[msg_type] += 1

    def run(self):
        """Main loop: receive messages until stopped."""
        self.connect()
        self.setup_files()

        # Request data streams from the autopilot — without this,
        # some messages may not be sent at useful rates
        self.request_streams()

        error_logger.debug(f"[*] Logging started. Press Ctrl+C to stop.")
        try:
            while self.keep_running:
                msg = self.conn.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                if msg.get_type() == "BAD_DATA":
                    continue
                self.log_message(msg)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def request_streams(self):
        """
        Request high-rate data streams from the autopilot.
        MAV_DATA_STREAM_RAW_SENSORS covers IMU and pressure;
        MAV_DATA_STREAM_POSITION covers GPS;
        MAV_DATA_STREAM_RC_CHANNELS covers servo outputs.
        """
        rate_hz = 50
        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
        ]
        for stream_id in streams:
            self.conn.mav.request_data_stream_send(
                self.conn.target_system,
                self.conn.target_component,
                stream_id,
                rate_hz,
                1  # 1 = start, 0 = stop
            )

    def shutdown(self):
        self.keep_running = False
        self.close_files()
        error_logger.info("\n[+] Shutdown summary:")
        for msg_name, count in self.message_counts.items():
            error_logger.info(f"    {msg_name}: {count} messages")


def main():
    parser = argparse.ArgumentParser(
        description="MAVLink telemetry logger for IDS dataset generation"
    )
    parser.add_argument(
        "--connection", default="udpin:127.0.0.1:14552",
        help="MAVLink connection string (default: udpin:127.0.0.1:14552)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/telemetry"),
        help="Output directory for CSV files (default: /tmp/telemetry)"
    )
    parser.add_argument(
        "--run-id", default=time.strftime("run_%Y%m%d_%H%M%S"),
        help="Run identifier used as CSV filename prefix"
    )
    args = parser.parse_args()

    logger = TelemetryLogger(args.output, args.run_id, args.connection)

    # Handle SIGTERM gracefully (for docker stop later)
    signal.signal(signal.SIGTERM, lambda *_: setattr(logger, "keep_running", False))

    logger.run()


if __name__ == "__main__":
    main()