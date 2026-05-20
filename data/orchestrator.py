#!/usr/bin/env python3
"""
orchestrator.py — Automated dataset generation.

Coordinates tcpdump, telemetry_logger, and ArduPilot SITL to generate
labeled dataset runs. Each run produces:
  - data/pcap/<run_id>.pcap
  - data/telemetry/run_<id>_<MSG_TYPE>.csv
  - data/labels/run_<id>.json

Usage:
    python3 orchestrator.py --manifest ./dataset/missions_manifest.csv --output ./dataset
"""

import argparse
import json
import logging
import signal
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime

from pymavlink import mavutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, output_dir: Path, manifest_path: Path,
                 connection: str):
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.connection = connection
        self.conn = None
        self.keep_running = True
 
        # Directories created by generate_missions.py already,
        # but ensure pcap/telemetry/phase_labels exist
        (output_dir / "pcap").mkdir(parents=True, exist_ok=True)
        (output_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        (output_dir / "phase_labels").mkdir(parents=True, exist_ok=True)

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

    def set_params(self, params: dict):
        """Send PARAM_SET for each drone parameter."""
        for param_id, value in params.items():
            log.info(f"  PARAM_SET {param_id} = {value}")
            self.conn.mav.param_set_send(
                self.conn.target_system,
                self.conn.target_component,
                param_id.encode("utf-8"),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            # Wait for PARAM_VALUE acknowledgment
            ack = self.conn.recv_match(
                type="PARAM_VALUE", blocking=True, timeout=5.0
            )
            if ack and ack.param_id.strip("\x00") == param_id:
                log.info(f"  {param_id} confirmed = {ack.param_value}")
            else:
                log.warning(f"  No confirmation for {param_id}")
 
    def upload_mission(self, waypoint_file: Path):
        """Parse QGC WPL 110 file and upload mission items via pymavlink."""
        items = []
        with open(waypoint_file, "r") as f:
            header = f.readline().strip()
            if header != "QGC WPL 110":
                raise ValueError(f"Bad waypoint header: {header}")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 12:
                    continue
                items.append(parts)
 
        n = len(items)
        log.info(f"  Uploading {n} mission items from {waypoint_file.name}")
 
        # Send mission count
        self.conn.mav.mission_count_send(
            self.conn.target_system,
            self.conn.target_component,
            n
        )
 
        # Respond to MISSION_REQUEST / MISSION_REQUEST_INT
        uploaded = set()
        deadline = time.time() + 30
        while len(uploaded) < n and time.time() < deadline:
            msg = self.conn.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                blocking=True, timeout=5.0
            )
            if msg is None:
                continue
            seq = msg.seq
            if seq >= n:
                break
            p = items[seq]
            self.conn.mav.mission_item_int_send(
                self.conn.target_system,
                self.conn.target_component,
                int(p[0]),              # seq
                int(p[2]),              # frame
                int(p[3]),              # command
                int(p[1]),              # current
                int(p[11]),             # autocontinue
                float(p[4]),            # param1
                float(p[5]),            # param2
                float(p[6]),            # param3
                float(p[7]),            # param4
                int(float(p[8]) * 1e7), # x (lat)
                int(float(p[9]) * 1e7), # y (lon)
                float(p[10])            # z (alt)
            )
            uploaded.add(seq)
 
        # Wait for MISSION_ACK
        ack = self.conn.recv_match(
            type="MISSION_ACK", blocking=True, timeout=10.0
        )
        if ack and ack.type == 0:
            log.info(f"  Mission upload accepted ({n} items)")
        else:
            log.warning(f"  Mission upload issue: {ack}")
 
    def wait_mission_complete(self, n_waypoints: int, timeout=900):
        """Wait for AUTO mission to complete, tracking waypoint progress."""
        log.info(f"  Waiting for mission complete ({n_waypoints} items)...")
        deadline = time.time() + timeout
        reached = set()
        last_seq = n_waypoints - 1  # 0-indexed, last item is RTL

        while time.time() < deadline and self.keep_running:
            msg = self.conn.recv_match(
                type=["STATUSTEXT", "MISSION_ITEM_REACHED", "HEARTBEAT"],
                blocking=True, timeout=2.0
            )
            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "MISSION_ITEM_REACHED":
                reached.add(msg.seq)
                log.info(f"  WP {msg.seq}/{last_seq} reached "
                         f"({len(reached)}/{n_waypoints})")

            elif mtype == "STATUSTEXT":
                text = msg.text.strip()
                if "Mission Complete" in text or "Auto disarmed" in text:
                    log.info(f"  Mission complete: {text}")
                    log.info(f"  Waypoints reached: {len(reached)}/{n_waypoints}")
                    return True, reached

            elif mtype == "HEARTBEAT":
                armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                if not armed and len(reached) > 0:
                    log.info("  Vehicle disarmed — mission done.")
                    return True, reached

        log.warning(f"  Timeout. Waypoints reached: {len(reached)}/{n_waypoints}")
        return False, reached
 
    def configure_wind(self, wind_params: dict, wind_direction_deg: int):
        """Placeholder: configure Gazebo wind plugin for this mission."""
        # TODO: Implement via gz service call or SDF patching
        log.info(f"  Wind: {wind_params}, direction={wind_direction_deg}°")
 
    def update_manifest_status(self, mission_id: str, new_status: str):
        """Update status column for mission_id in manifest CSV."""
        rows = []
        with open(self.manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["mission_id"] == mission_id:
                    row["status"] = new_status
                rows.append(row)
        with open(self.manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
 
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
                    attack_type: str = "none", waypoints_reached: list = None,
                    mission_success: bool = False):
        label = {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": end_time - start_time,
            "mission_success": mission_success,
            "waypoints_reached": waypoints_reached or [],
            "attack_type": attack_type,
            "attack_start": None,
            "attack_end": None,
            "notes": ""
        }
        label_path = self.output_dir / "labels" / f"{run_id}.json"
        with open(label_path, "w") as f:
            json.dump(label, f, indent=2)
        log.info(f"[+] Label written -> {label_path}")
 
    def run_single(self, mission: dict):
        mission_id = mission["mission_id"]
        log.info(f"=== Starting {mission_id} ({mission['profile']}) ===")
 
        # Load full params from meta JSON
        meta_path = self.output_dir / "meta" / f"{mission_id}_params.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
 
        # Set drone parameters
        self.set_params(meta["drone_params"])
 
        # Configure wind
        self.configure_wind(meta["wind_params"], meta["wind_direction_deg"])
 
        # Upload mission waypoints
        wp_path = self.output_dir / "waypoints" / mission["waypoint_file"]
        self.upload_mission(wp_path)
 
        # Start capture
        tcpdump_proc = self.start_tcpdump(mission_id)
        telemetry_proc = self.start_telemetry_logger(mission_id)
        time.sleep(1)
 
        start_time = time.time()
        try:
            self.wait_for_ready()
            self.set_mode("AUTO")
            self.arm()
            n_wps = len(open(wp_path).readlines()) - 1  # minus header line
            success, reached_wps = self.wait_mission_complete(n_wps, timeout=900)
            self.write_label(mission_id, start_time, end_time,
                        waypoints_reached=list(reached_wps),
                        mission_success=success)
            if not success:
                log.warning(f"  Mission {mission_id} did not complete cleanly")
        except Exception as e:
            log.error(f"Run {mission_id} failed: {e}")
        finally:
            end_time = time.time()
            tcpdump_proc.terminate()
            tcpdump_proc.wait()
            telemetry_proc.terminate()
            telemetry_proc.wait()
            self.write_label(mission_id, start_time, end_time)
            self.update_manifest_status(mission_id, "completed")
            log.info(f"=== {mission_id} complete ({end_time - start_time:.1f}s) ===")

    def run(self):
        """Iterate manifest, skip completed, run pending missions."""
        self.connect()
 
        with open(self.manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            missions = [row for row in reader]
 
        pending = [m for m in missions if m["status"] == "pending"]
        total = len(missions)
        done = total - len(pending)
        log.info(f"Manifest: {total} missions, {done} completed, {len(pending)} pending")
 
        for i, mission in enumerate(pending):
            if not self.keep_running:
                log.info("Interrupted — exiting.")
                break
            log.info(f"[{done + i + 1}/{total}] Next: {mission['mission_id']}")
            self.run_single(mission)
            time.sleep(2)  # brief pause between missions
 
        log.info("All pending missions processed.")


def main():
    parser = argparse.ArgumentParser(
        description="UAV IDS dataset generation orchestrator"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to missions_manifest.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./dataset"),
        help="Dataset root directory (default: ./dataset)"
    )
    parser.add_argument(
        "--connection", default="udpin:127.0.0.1:14552",
        help="MAVLink connection string (default: udpin:127.0.0.1:14552)"
    )
    args = parser.parse_args()
 
    orchestrator = Orchestrator(
        output_dir=args.output,
        manifest_path=args.manifest,
        connection=args.connection
    )
 
    signal.signal(
        signal.SIGTERM,
        lambda *_: setattr(orchestrator, "keep_running", False)
    )
    signal.signal(
        signal.SIGINT,
        lambda *_: setattr(orchestrator, "keep_running", False)
    )
 
    orchestrator.run()

if __name__ == "__main__":
    main()
