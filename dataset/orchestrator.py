#!/usr/bin/env python3
"""
orchestrator.py — Automated dataset generation.

Coordinates tcpdump, telemetry_logger, and ArduPilot SITL to generate
labeled dataset runs. Each run produces:
  - data/pcap/<run_id>.pcap
  - data/telemetry/run_<id>_<MSG_TYPE>.csv
  - data/labels/run_<id>.json

Usage:
    python3 orchestrator.py --runs 10 --output /data --takeoff-alt 10
"""

import argparse
import json
import logging
import signal
import subprocess
import time
from pathlib import Path
from datetime import datetime

from pymavlink import mavutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, output_dir: Path, runs: int, takeoff_alt: float,
                 connection: str):
        self.output_dir = output_dir
        self.runs = runs
        self.takeoff_alt = takeoff_alt
        self.connection = connection
        self.conn = None
        self.keep_running = True

        # Create output directories
        (output_dir / "pcap").mkdir(parents=True, exist_ok=True)
        (output_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def connect(self):
        log.info(f"[+] Connecting to {self.connection}...")
        self.conn = mavutil.mavlink_connection(self.connection)
        self.conn.wait_heartbeat()
        log.info(f"Heartbeat from system {self.conn.target_system}:"
                 f"{self.conn.target_component}")

    def wait_for_ready(self, timeout=60):
        """Wait until GPS has fix."""
        log.info("[+] Waiting for GPS fix...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="GPS_RAW_INT", blocking=True, timeout=2.0
            )
            if msg and msg.fix_type >= 3:
                log.info("GPS fix acquired.")
                return True
        log.warning("[?] Timeout waiting for GPS fix — proceeding anyway.")
        return False

    def arm(self):
        log.info("Arming...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

    def disarm(self):
        log.info("Disarming...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def set_mode(self, mode: str):
        mode_id = self.conn.mode_mapping()[mode]
        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        log.info(f"[+] Mode set to {mode}")

    def takeoff(self, alt: float):
        log.info(f"[+] Taking off to {alt}m...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, alt
        )
        self._wait_altitude(alt, tolerance=1.0, timeout=30)

    def land(self):
        log.info("[+] Landing...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self._wait_disarmed(timeout=30)

    def hover(self, duration: float):
        log.info(f"[+] Hovering for {duration}s...")
        time.sleep(duration)

    def _wait_for_ack(self, command, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="COMMAND_ACK", blocking=True, timeout=2.0
            )
            if msg and msg.command == command:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return True
                else:
                    log.warning(f"Command {command} rejected: {msg.result}")
                    return False
        log.warning(f"[+] Timeout waiting for ACK for command {command}")
        return False

    def _wait_altitude(self, target_alt: float, tolerance=1.0, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0
            )
            if msg is None:
                continue
            current_alt = msg.relative_alt / 1000.0
            if abs(current_alt - target_alt) < tolerance:
                log.info(f"[+] Target altitude {target_alt}m reached.")
                return
        log.warning(f"[?] Timeout waiting for altitude {target_alt}m")

    def _wait_disarmed(self, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="HEARTBEAT", blocking=True, timeout=2.0
            )
            if msg is None:
                continue
            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if not armed:
                log.info("[+] Disarmed.")
                return
        log.warning("[?] Timeout waiting for disarm")

    def start_tcpdump(self, run_id: str) -> subprocess.Popen:
        pcap_path = self.output_dir / "pcap" / f"{run_id}.pcap"
        log.info(f"[+] Starting tcpdump -> {pcap_path}")
        return subprocess.Popen(
            ["tcpdump", "-i", "hwsim0", "-w", str(pcap_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def start_telemetry_logger(self, run_id: str) -> subprocess.Popen:
        telemetry_dir = self.output_dir / "telemetry"
        log.info(f"[+] Starting telemetry logger -> {telemetry_dir}")
        return subprocess.Popen(
            [
                "python3", "telemetry_logger.py",
                "--connection", self.connection,
                "--output", str(telemetry_dir),
                "--run-id", run_id
            ]
        )

    def write_label(self, run_id: str, start_time: float, end_time: float,
                    attack_type: str = "none"):
        label = {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": end_time - start_time,
            "attack_type": attack_type,
            "attack_start": None,
            "attack_end": None,
            "notes": ""
        }
        label_path = self.output_dir / "labels" / f"{run_id}.json"
        with open(label_path, "w") as f:
            json.dump(label, f, indent=2)
        log.info(f"[+] Label written -> {label_path}")

    def run_single(self, run_index: int):
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_index:03d}"
        log.info(f"[+] === Starting {run_id} ===")

        tcpdump_proc = self.start_tcpdump(run_id)
        telemetry_proc = self.start_telemetry_logger(run_id)
        time.sleep(1)  # give subprocesses time to start

        start_time = time.time()
        try:
            self.wait_for_ready()
            self.set_mode("GUIDED")
            self.arm()
            self.takeoff(self.takeoff_alt)
            self.hover(30)  # 30s hover — TODO: replace with QGC mission
            self.land()
        except Exception as e:
            log.error(f"[!] Run {run_id} failed: {e}")
        finally:
            end_time = time.time()
            tcpdump_proc.terminate()
            tcpdump_proc.wait()
            telemetry_proc.terminate()
            telemetry_proc.wait()
            self.write_label(run_id, start_time, end_time, attack_type="none")
            log.info(f"=== {run_id} complete ({end_time - start_time:.1f}s) ===")

    def run(self):
        self.connect()
        for i in range(self.runs):
            if not self.keep_running:
                break
            self.run_single(i)
            if i < self.runs - 1:
                log.info("[+] Waiting 5s before next run...")
                time.sleep(5)
        log.info("[+] All runs complete.")


def main():
    parser = argparse.ArgumentParser(
        description="UAV IDS dataset generation orchestrator"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs to execute (default: 1)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/data"),
        help="Output directory for dataset (default: /data)"
    )
    parser.add_argument(
        "--takeoff-alt", type=float, default=10.0,
        help="Takeoff altitude in meters (default: 10.0)"
    )
    parser.add_argument(
        "--connection", default="udpin:127.0.0.1:14552",
        help="MAVLink connection string (default: udpin:127.0.0.1:14552)"
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(
        output_dir=args.output,
        runs=args.runs,
        takeoff_alt=args.takeoff_alt,
        connection=args.connection
    )

    signal.signal(
        signal.SIGTERM,
        lambda *_: setattr(orchestrator, "keep_running", False)
    )

    orchestrator.run()


if __name__ == "__main__":
    main()