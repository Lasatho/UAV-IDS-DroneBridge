#!/usr/bin/env python3
"""
orchestrator.py — Automated dataset generation.

Coordinates tcpdump, telemetry_logger, and ArduPilot SITL to generate
labeled dataset runs. Each run produces:
  - data/pcap/<mission_id>.pcap
  - data/telemetry/<mission_id>_<MSG_TYPE>.csv
  - data/phase_labels/<mission_id>.json

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
import os
from pathlib import Path
from datetime import datetime
import math
import random

from pymavlink import mavutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Push notifications via ntfy.sh (optional)
# Set env var NTFY_TOPIC to enable. Install ntfy app on phone,
# subscribe to the same topic.
# ---------------------------------------------------------------------------
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")


def notify(message: str):
    """Send push notification via ntfy.sh. Fails silently."""
    if not NTFY_TOPIC:
        return
    try:
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=message.encode(),
                method="POST"
            ), timeout=5
        )
    except Exception:
        pass


class Orchestrator:
    def __init__(self, output_dir: Path, manifest_path: Path,
                 connection: str):
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.connection = connection
        self.conn = None
        self.keep_running = True
        self.consecutive_failures = 0

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

    def wait_for_ready(self, timeout=120):
        """Wait until GPS has fix."""
        log.info("[+] Waiting for GPS fix...")
        while self.conn.recv_match(blocking=False) is not None:
            pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="GPS_RAW_INT", blocking=True, timeout=2.0
            )
            if msg and msg.fix_type >= 3:
                log.info("GPS fix acquired.")
                return True
        log.warning("[?] Timeout waiting for GPS fix.")
        return False

    def arm(self):
        log.info("Arming...")
        while self.conn.recv_match(blocking=False) is not None:
            pass
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0  # param2=21196 = force arm
        )
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

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
        while self.conn.recv_match(blocking=False) is not None:
            pass
        for param_id, value in params.items():
            log.info(f"  PARAM_SET {param_id} = {value}")
            self.conn.mav.param_set_send(
                self.conn.target_system,
                self.conn.target_component,
                param_id.encode("utf-8"),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
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

        self.conn.mav.mission_count_send(
            self.conn.target_system,
            self.conn.target_component,
            n
        )

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

        ack = self.conn.recv_match(
            type="MISSION_ACK", blocking=True, timeout=10.0
        )
        if ack and ack.type == 0:
            log.info(f"  Mission upload accepted ({n} items)")
        else:
            log.warning(f"  Mission upload issue: {ack}")

    def estimate_mission_timeout(self, wp_path: Path, meta: dict) -> int:
        """Fuck this. Fixed 30 minute timeout for all missions."""
        return 1800
        # """Estimate max mission duration from actualy waypoint distances"""
        # wps = []
        # with open(wp_path, "r") as f:
        #     f.readline()    # skip header
        #     for line in f:
        #         parts = line.strip().split("\t")
        #         if len(parts) < 12:
        #             continue
        #     cmd = int(parts[3])
        #     if cmd in (16,22):
        #         lat = float(parts[8])
        #         lon = float(parts[9])
        #         alt = float(parts[10])
        #         if lat != 0.0 or lon != 0.0:
        #             wps.append((lat, lon, alt))

        # # calculate total distance between consecutive waypoints
        # total_dist = 0.0
        # R = 6378137.0
        # for i in range(1, len(wps)):
        #     dlat = math.radians(wps[i][0] - wps[i-1][0])
        #     dlon = math.radians(wps[i][1] - wps[i-1][1])
        #     a = (math.sin(dlat/2)**2 + math.cos(math.radians(wps[i-1][0])) * math.cos(math.radians(wps[i][0])) * math.sin(dlon/2)**2)
        #     total_dist += R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        # speed_cms = meta["drone_params"].get("WPNAV_SPEED", 500)
        # speed_ms = speed_cms / 100.0

        # # RTL overhead
        # rtl_alt_cm = meta["drone_params"].get("RTL_ALT", 3000)
        # speed_up_cms = meta["drone_params"].get("WPNAV_SPEED_UP", 250)
        # speed_dn_cms = meta["drone_params"].get("WPNAV_SPEED_DN", 150)
        # rtl_overhead_s= (rtl_alt_cm  / speed_up_cms) + (rtl_alt_cm / speed_dn_cms) + 60

        # flight_s = total_dist / max(speed_ms, 1.0)
        # estimated_s = flight_s + rtl_overhead_s
        # timeout = max(300, int(estimated_s * 2.5))
        # log.info(f"  Mission timeout: {timeout}s (flight={flight_s:.0f}s, "
        #          f"dist={total_dist:.0f}m, rtl={rtl_overhead_s:.0f}s, {speed_ms:.1f} m/s)")
        # return timeout

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
        self._wait_altitude(alt, tolerance=1.0, timeout=60)

    def land(self):
        log.info("[+] Landing...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self._wait_disarmed(timeout=60)

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

    def _wait_altitude(self, target_alt: float, tolerance=1.0, timeout=60):
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
                return True
        log.warning(f"[?] Timeout waiting for altitude {target_alt}m")
        return False

    def _wait_disarmed(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="HEARTBEAT", blocking=True, timeout=2.0
            )
            if msg is None:
                continue
            if msg.type != mavutil.mavlink.MAV_TYPE_QUADROTOR:
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
                "--connection", "udpin:127.0.0.1:14553",
                "--output", str(telemetry_dir),
                "--run-id", run_id
            ]
        )

    def start_attack(self, attack_type: str, mission_id: str) -> subprocess.Popen:
        """Start an attack script as a subprocess."""
        script = Path(f"attacks/{attack_type}.py")
        log.info(f"[!] Starting attack: {attack_type}")
        return subprocess.Popen(
            ["python3", str(script), "--mission-id", mission_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    
    def wait_mission_complete(self, timeout=900, attack_type="none",
                              attack_start_offset=None, attack_duration=None,
                              mission_id=None):
        log.info(f"  Waiting for mission complete (timeout={timeout}s)...")
        deadline = time.time() + timeout
        min_flight_time = time.time() + 60
        reached = set()
        first_wp_time = None

        # Attack state
        attack_proc = None
        attack_started = False
        attack_start_ts = None
        attack_end_ts = None
        attack_ready = False  # True once first WP reached + 60s flying

        while time.time() < deadline and self.keep_running:
            msg = self.conn.recv_match(
                type=["STATUSTEXT", "MISSION_ITEM_REACHED", "HEARTBEAT"],
                blocking=True, timeout=2.0
            )

            # Check if attack conditions are met
            if (not attack_ready and attack_type != "none"
                    and first_wp_time is not None
                    and time.time() > min_flight_time
                    and time.time() - first_wp_time > 10):
                attack_ready = True
                log.info(f"  Attack window open (first WP + 60s flying)")

            # Start attack at randomized offset
            if (attack_ready and not attack_started
                    and attack_start_offset is not None
                    and time.time() - first_wp_time >= attack_start_offset):
                attack_proc = self.start_attack(attack_type, mission_id)
                attack_start_ts = time.time()
                attack_started = True
                log.info(f"  [!] Attack {attack_type} STARTED at offset "
                         f"{time.time() - first_wp_time:.0f}s")

            # Stop attack after duration
            if (attack_started and attack_proc is not None
                    and attack_end_ts is None
                    and time.time() - attack_start_ts >= attack_duration):
                attack_proc.terminate()
                attack_proc.wait()
                attack_end_ts = time.time()
                log.info(f"  [!] Attack {attack_type} STOPPED after "
                         f"{attack_duration:.0f}s")

            if msg is None:
                continue

            mtype = msg.get_type()

            if mtype == "MISSION_ITEM_REACHED":
                reached.add(msg.seq)
                if first_wp_time is None:
                    first_wp_time = time.time()
                log.info(f"  WP {msg.seq} reached ({len(reached)} total)")

            elif mtype == "STATUSTEXT":
                text = msg.text.strip()
                log.info(f"  STATUSTEXT: {text}")
                if ("Mission Complete" in text or
                    "Auto disarmed" in text or
                    "Disarming motors" in text):
                    log.info(f"  Mission complete! WPs reached: {len(reached)}")
                    # Clean up attack if still running
                    if attack_proc is not None and attack_end_ts is None:
                        attack_proc.terminate()
                        attack_proc.wait()
                        attack_end_ts = time.time()
                    return True, reached, attack_start_ts, attack_end_ts

            elif mtype == "HEARTBEAT":
                if msg.type != mavutil.mavlink.MAV_TYPE_QUADROTOR:
                    continue
                armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                if not armed and len(reached) > 0 and time.time() > min_flight_time:
                    log.info(f"  Vehicle disarmed — mission done. "
                             f"WPs reached: {len(reached)}")
                    if attack_proc is not None and attack_end_ts is None:
                        attack_proc.terminate()
                        attack_proc.wait()
                        attack_end_ts = time.time()
                    return True, reached, attack_start_ts, attack_end_ts

        # Timeout — clean up attack
        if attack_proc is not None and attack_end_ts is None:
            attack_proc.terminate()
            attack_proc.wait()
            attack_end_ts = time.time()

        log.warning(f"  Timeout. Waypoints reached: {len(reached)}")
        return False, reached, attack_start_ts, attack_end_ts
    
    def write_label(self, run_id: str, start_time: float, end_time: float,
                    attack_type: str = "none", attack_start: float = None,
                    attack_end: float = None, waypoints_reached: list = None,
                    mission_success: bool = False):
        label = {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": end_time - start_time,
            "mission_success": mission_success,
            "waypoints_reached": waypoints_reached or [],
            "attack_type": attack_type,
            "attack_start": attack_start,
            "attack_end": attack_end,
            "attack_duration": (attack_end - attack_start) if (attack_start and attack_end) else None,
            "notes": ""
        }
        label_path = self.output_dir / "phase_labels" / f"{run_id}.json"
        with open(label_path, "w") as f:
            json.dump(label, f, indent=2)
        log.info(f"[+] Label written -> {label_path}")

    def wait_ekf_ready(self, timeout=30):
        """Wait until EKF has converged."""
        log.info("[+] Waiting for EKF convergence...")
        while self.conn.recv_match(blocking=False) is not None:
            pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="EKF_STATUS_REPORT", blocking=True, timeout=2.0
            )
            if msg:
                if msg.flags & 0x0F == 0x0F:
                    log.info("EKF converged.")
                    return True
        log.warning("[?] EKF not converged — trying anyway.")
        return False

    def reboot_sitl(self):
        """Send preflight reboot to reset SITL state."""
        log.info("[+] Rebooting SITL...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        time.sleep(20)
        self.conn = mavutil.mavlink_connection(self.connection)
        self.conn.wait_heartbeat()
        log.info("[+] SITL rebooted, heartbeat OK.")

    def verify_armed(self, timeout=10):
        """Check heartbeat to verify armed state."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(type="HEARTBEAT", blocking=True, timeout=2.0)
            if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                log.info("  Armed confirmed via heartbeat.")
                return True
        log.warning("  Vehicle NOT armed.")
        return False

    def verify_flying(self, timeout=30):
        """Check if vehicle has left the ground."""
        log.info("  Verifying takeoff...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0
            )
            if msg and msg.relative_alt > 1000:
                log.info(f"  Airborne (alt={msg.relative_alt/1000:.1f}m)")
                return True
        log.warning("  Vehicle not airborne.")
        return False

    def run_single(self, mission: dict):
        mission_id = mission["mission_id"]
        log.info(f"=== Starting {mission_id} ({mission['profile']}) ===")
        wp_path = self.output_dir / "waypoints" / mission["waypoint_file"]

        # Load full params from meta JSON
        meta_path = self.output_dir / "meta" / f"{mission_id}_params.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Configure wind
        self.configure_wind(meta["wind_params"], meta["wind_direction_deg"])

        # Start capture
        tcpdump_proc = self.start_tcpdump(mission_id)
        telemetry_proc = self.start_telemetry_logger(mission_id)
        time.sleep(1)

        start_time = time.time()
        success = False
        reached_wps = set()

        atk_start = None
        atk_end = None
            
        try:
            if not self.wait_for_ready():
                raise RuntimeError("No GPS fix")
            self.wait_ekf_ready(timeout=30)

            # Set params and upload AFTER SITL is ready
            self.set_params(meta["drone_params"])
            self.upload_mission(wp_path)

            self.set_mode("GUIDED")
            if not self.arm():
                raise RuntimeError("Arming failed")
            if not self.verify_armed():
                raise RuntimeError("Vehicle not armed")
            # Takeoff in GUIDED first to prevent auto-disarm
            self.takeoff(15)
            if not self.verify_flying():
                raise RuntimeError("Takeoff failed")
            # Now switch to AUTO — mission continues from WP2
            self.set_mode("AUTO")

            # Attack scheduling
            attack_type = meta.get("attack_type", "none")
            attack_start_offset = None
            attack_duration = None

            if attack_type != "none":
                # Randomized: start 60-180s after first WP, duration 15-60s
                attack_start_offset = random.uniform(60, 180)
                attack_duration = random.uniform(15, 60)
                log.info(f"  Attack scheduled: {attack_type}, "
                         f"offset={attack_start_offset:.0f}s, "
                         f"duration={attack_duration:.0f}s")

            mission_timeout = self.estimate_mission_timeout(wp_path, meta)
            success, reached_wps, atk_start, atk_end = self.wait_mission_complete(
                timeout=mission_timeout,
                attack_type=attack_type,
                attack_start_offset=attack_start_offset,
                attack_duration=attack_duration,
                mission_id=mission_id
            )

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
            self.write_label(mission_id, start_time, end_time,
                             attack_type=meta.get("attack_type", "none"),
                             attack_start=atk_start,
                             attack_end=atk_end,
                             waypoints_reached=list(reached_wps),
                             mission_success=success)
            status = "completed" if success else "failed"
            self.update_manifest_status(mission_id, status)
            log.info(f"=== {mission_id} {status} ({end_time - start_time:.1f}s) ===")

            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                notify(f"[UAV-IDS] {mission_id} FAILED: "
                       f"{self.consecutive_failures} consecutive failures")
                if self.consecutive_failures >= 5:
                    notify("[UAV-IDS] 5 consecutive failures — stopping orchestrator.")
                    log.error("5 consecutive failures — stopping.")
                    self.keep_running = False
                    (self.output_dir / ".stop_orchestrator").touch()

            self.reboot_sitl()

    def run(self):
        """Iterate manifest, skip completed/failed, run pending missions."""
        
        stop_flag = self.output_dir / ".stop_orchestrator"
        if stop_flag.exists():
            log.error("Stop flag found — previous run had 5 consecutive failures. "
                      "Delete .stop_orchestrator to resume.")
            notify("[UAV-IDS] Stop flag present — orchestrator not starting.")
            return
        
        self.connect()

        with open(self.manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            missions = [row for row in reader]

        pending = [m for m in missions if m["status"] == "pending"]
        total = len(missions)
        done = total - len(pending)
        log.info(f"Manifest: {total} missions, {done} completed, "
                 f"{len(pending)} pending")

        notify(f"[UAV-IDS] Orchestrator started: {len(pending)} pending missions")

        completed_count = 0
        failed_count = 0
        for i, mission in enumerate(pending):
            if not self.keep_running:
                log.info("Interrupted — exiting.")
                notify(f"[UAV-IDS] Interrupted after {completed_count} completed, "
                       f"{failed_count} failed")
                break
            log.info(f"[{done + i + 1}/{total}] Next: {mission['mission_id']}")
            self.run_single(mission)

            # Track counts
            if self.consecutive_failures == 0:
                completed_count += 1
            else:
                failed_count += 1

            # Progress notification every 100 missions
            processed = completed_count + failed_count
            if processed % 100 == 0:
                notify(f"[UAV-IDS] Progress: {processed}/{len(pending)} processed "
                       f"({completed_count} ok, {failed_count} failed)")

            time.sleep(2)

        notify(f"[UAV-IDS] Orchestrator finished: {completed_count} completed, "
               f"{failed_count} failed")
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