#!/usr/bin/env python3
"""
orchestrator.py — Automated dataset generation.

Coordinates tcpdump, telemetry_logger, and ArduPilot SITL to generate
labeled dataset runs. Each run produces:
  - data/pcap/<mission_id>.pcap
  - data/telemetry/<mission_id>_<MSG_TYPE>.csv
  - data/phase_labels/<mission_id>.json
  - data/wind/<mission_id>_wind.csv (commanded wind series)

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
import threading

from pymavlink import mavutil

from wind_model import WindModel

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

# Attack runs keep recording this long after the attack has ended, even if
# the vehicle disarmed (landed, crashed, terminated) during the attack: the
# post-attack phase is part of the attack sample. 30 s covers a LAND descent
# from 15 m and holds at least one full 16 s window (64 samples at 4 Hz).
ATTACK_POST_WINDOW_S = 30.0

# Gazebo model and its spawn pose from the world SDF (iris_runway.sdf:
# <pose degrees="true">0 0 0.195 0 0 90</pose>); yaw 90 deg as quaternion.
GZ_WORLD = "iris_runway"
GZ_MODEL = "iris_with_gimbal"
GZ_SPAWN_POSE = ("position: {x: 0, y: 0, z: 0.195}, "
                 "orientation: {x: 0, y: 0, z: 0.7071068, w: 0.7071068}")

# Wind: the orchestrator publishes a time-varying mean wind vector
# (wind_model.py) at WIND_RATE_HZ during the flight and zero wind before and
# after it. The WindEffects low-pass (time_for_rise in the world SDF) is set
# to one update period so the published series reaches the model unchanged.
WIND_TOPIC = f"/world/{GZ_WORLD}/wind"
WIND_RATE_HZ = 4.0

# A normal flight that crashes (ArduCopter "Crash: Disarming", e.g. tipped
# over by a gust at touchdown) is no benign data, and with the fixed wind seed
# every retry crashes the same way. It is flown once more with the seed
# offset by WIND_RESEED_OFFSET; the crashed run is moved to CRASHED_RUNS_DIR.
WIND_RESEED_OFFSET = 100000
CRASHED_RUNS_DIR = "_crashed_runs"

# Between missions the orchestrator holds while this flag exists, so the
# watchdog can restart gazebo+ardupilot without hitting a pose reset or SITL
# reboot. After the flag is removed the SITL is rebooted again.
PAUSE_FLAG = ".pause_orchestrator"

# WORKAROUND (2026-10-08): with Gazebo wind the landed model keeps being
# pushed, ArduCopter's land detector never triggers and the vehicle stays
# armed on the ground until the mission timeout (75 min for normal runs).
# The orchestrator treats the vehicle as landed after LANDED_HOLD_S of
# relative altitude < LANDED_ALT_M and near-zero velocity (after having been
# above AIRBORNE_ALT_M), force-disarms it and marks this in the label.
LANDED_ALT_M = 0.5
LANDED_VH_MS = 0.3
LANDED_VZ_MS = 0.2
LANDED_HOLD_S = 10.0
AIRBORNE_ALT_M = 5.0

# Child processes (tcpdump, telemetry logger, attack scripts) get this long
# to exit after SIGTERM before they are killed.
CHILD_STOP_TIMEOUT_S = 15.0


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


def stop_process(proc: subprocess.Popen, name: str):
    """SIGTERM, then SIGKILL if the process does not exit in time.

    A child that hangs (e.g. blocked waiting for a MAVLink link that is
    gone) must not block the whole dataset run.
    """
    proc.terminate()
    try:
        proc.wait(timeout=CHILD_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.warning(f"  {name} did not exit after SIGTERM — killing")
        proc.kill()
        proc.wait()


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
        (output_dir / "wind").mkdir(parents=True, exist_ok=True)

    def _bind_vehicle_heartbeat(self, timeout=60):
        """Bind the connection target to the real ArduPilot vehicle.

        MAVProxy emits a phantom HEARTBEAT with system id 0; a plain
        wait_heartbeat() may latch onto that, leaving target_system=0 so
        every PARAM_SET / takeoff command is sent into the void and the
        mission silently fails at takeoff. Skip sysid 0 and match the
        quadrotor autopilot explicitly.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(type="HEARTBEAT", blocking=True,
                                       timeout=5.0)
            if msg is None:
                continue
            src = msg.get_srcSystem()
            if src == 0 or msg.type != mavutil.mavlink.MAV_TYPE_QUADROTOR:
                continue
            self.conn.target_system = src
            self.conn.target_component = msg.get_srcComponent()
            return True
        return False

    def connect(self):
        log.info(f"[+] Connecting to {self.connection}...")
        self.conn = mavutil.mavlink_connection(self.connection)
        if not self._bind_vehicle_heartbeat():
            raise RuntimeError("No vehicle heartbeat (only phantom sysid 0?)")
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
        """Send PARAM_SET for each drone parameter and require confirmation.

        The autopilot answers with PARAM_VALUE carrying the new value.
        Unknown parameter names are ignored silently by ArduPilot (no
        PARAM_VALUE) — this is how the v1 presets (WPNAV_*, renamed in
        ArduCopter 4.8) never took effect. A parameter that is not
        confirmed with the requested value fails the run.
        """
        while self.conn.recv_match(blocking=False) is not None:
            pass
        for param_id, value in params.items():
            confirmed = None
            for attempt in range(3):
                log.info(f"  PARAM_SET {param_id} = {value}")
                self.conn.mav.param_set_send(
                    self.conn.target_system,
                    self.conn.target_component,
                    param_id.encode("utf-8"),
                    float(value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                )
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    ack = self.conn.recv_match(
                        type="PARAM_VALUE", blocking=True, timeout=1.0
                    )
                    if ack and ack.param_id.strip("\x00") == param_id:
                        confirmed = ack.param_value
                        break
                if confirmed is not None:
                    break
            if confirmed is None or abs(confirmed - float(value)) > 1e-3 * max(1.0, abs(float(value))):
                raise RuntimeError(f"Parameter {param_id}={value} not confirmed "
                                   f"(got {confirmed})")
            log.info(f"  {param_id} confirmed = {confirmed}")

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
        self.mission_count = n
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
        """Fixed timeout: 10 min for attack runs (failsafe/hold is the
        expected attack effect), 75 min for normal missions (longest v2
        mission, conservative preset at 3 m/s, is estimated at ~60 min)."""
        if meta.get("attack_type", "none") != "none":
            return 600
        return 4500
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

    def _wind_publisher(self):
        """Lazily advertise the WindEffects topic (gz-transport, in-process).

        The gz CLI needs ~2 s per call, too slow for gusts. Raises if no
        subscriber (Gazebo) shows up: flying without the configured wind
        would silently produce mislabeled data.
        """
        if getattr(self, "_wind_pub", None) is None:
            from gz.transport13 import Node
            from gz.msgs10.wind_pb2 import Wind
            self._wind_node = Node()
            self._wind_msg_type = Wind
            self._wind_pub = self._wind_node.advertise(WIND_TOPIC, Wind)
        for _ in range(50):
            if self._wind_pub.has_connections():
                return self._wind_pub
            time.sleep(0.1)
        raise RuntimeError(f"No subscriber on {WIND_TOPIC} (Gazebo down?)")

    def publish_wind(self, vx: float, vy: float):
        """World frame ENU (x=East, y=North, z=Up)."""
        msg = self._wind_msg_type()
        msg.linear_velocity.x = vx
        msg.linear_velocity.y = vy
        msg.linear_velocity.z = 0.0
        msg.enable_wind = True
        self._wind_pub.publish(msg)

    def zero_wind(self):
        """Calm air before takeoff and after the flight (a leftover wind
        pushes the landed model and the next mission's ground phase)."""
        self._wind_publisher()
        for _ in range(3):
            self.publish_wind(0.0, 0.0)
            time.sleep(1 / WIND_RATE_HZ)

    def start_wind(self, mission_id: str, wind_params: dict,
                   wind_direction_deg: int, seed: int = None) -> dict:
        """Start publishing the mission's time-varying wind in a thread.

        Seeded from the mission number unless a seed is given (reproducible).
        The commanded series is written to <output>/wind/<mission_id>_wind.csv.
        """
        self._wind_publisher()
        if seed is None:
            seed = int(mission_id.split("_")[-1])
        model = WindModel(wind_params, wind_direction_deg, seed)
        path = self.output_dir / "wind" / f"{mission_id}_wind.csv"
        stop = threading.Event()

        def loop():
            dt = 1 / WIND_RATE_HZ
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["host_time", "speed", "direction_deg", "gust",
                            "v_east", "v_north"])
                nxt = time.time()
                while not stop.is_set():
                    speed, direction, gust = model.step(dt)
                    vx, vy = WindModel.to_enu(speed, direction)
                    self.publish_wind(vx, vy)
                    w.writerow([f"{time.time():.3f}", f"{speed:.3f}",
                                f"{direction:.1f}", f"{gust:.3f}",
                                f"{vx:.3f}", f"{vy:.3f}"])
                    nxt += dt
                    stop.wait(max(0.0, nxt - time.time()))

        self._wind_stop = stop
        self._wind_thread = threading.Thread(target=loop, daemon=True)
        self._wind_thread.start()
        log.info(f"  Wind: {model.turbulence}, {model.vmin:.1f}-{model.vmax:.1f} m/s "
                 f"from {wind_direction_deg}° (seed={seed}) -> {path.name}")
        return {**model.describe(), "seed": seed, "file": f"wind/{path.name}"}

    def stop_wind(self):
        """Stop the wind thread (if any) and set calm air."""
        if getattr(self, "_wind_thread", None) is not None:
            self._wind_stop.set()
            self._wind_thread.join(timeout=5)
            self._wind_thread = None
        self.zero_wind()

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

    def start_attack(self, attack_type: str, mission_id: str,
                     attack_variant: str = None) -> subprocess.Popen:
        """Start an attack script as a subprocess."""
        script = Path(f"attacks/{attack_type}.py")
        cmd = ["python3", str(script), "--mission-id", mission_id]
        if attack_variant:
            cmd += ["--variant", attack_variant]
        log.info(f"[!] Starting attack: {attack_type}"
                 + (f" ({attack_variant})" if attack_variant else ""))
        # stderr is not read: a PIPE would fill up and block the script.
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def wait_mission_complete(self, timeout=900, attack_type="none",
                              attack_start_offset=None, attack_duration=None,
                              mission_id=None, attack_variant=None) -> dict:
        """Monitor the flight until it is over; run the attack in between.

        Normal runs end at the first sign the flight is over ("Mission
        Complete", a disarm STATUSTEXT, or a disarmed heartbeat).

        Attack runs: the attack always runs for its scheduled duration, and
        recording continues until ATTACK_POST_WINDOW_S after the attack has
        ended, even if the vehicle disarmed earlier (LAND/crash/flight
        termination are attack effects, not the end of the sample). The run
        then ends once the flight is over, or at the timeout.

        Returns a dict with mission_complete (the last mission item was
        reached or "Mission Complete" seen; a disarm alone does not count),
        reached waypoints, attack start/end, disarm time/reason, flight mode
        changes and all STATUSTEXTs.

        ArduCopter 4.8 sends no "Mission Complete" for missions ending in an
        RTL item; MISSION_ITEM_REACHED of that last item arrives on landing.
        """
        log.info(f"  Waiting for mission complete (timeout={timeout}s)...")
        deadline = time.time() + timeout
        min_flight_time = time.time() + 60
        reached = set()
        first_wp_time = None

        # Flight state
        mission_complete = False
        last_seq = getattr(self, "mission_count", 0) - 1
        disarm_time = None
        disarm_reason = None
        mode = None
        mode_changes = []
        statustext = []
        was_airborne = False
        on_ground_since = None
        landed_by_orchestrator = False

        # Attack state
        attack_proc = None
        attack_started = False
        attack_start_ts = None
        attack_end_ts = None
        attack_ready = False  # True once first WP reached + 60s flying

        def flight_over() -> bool:
            return mission_complete or disarm_time is not None

        def run_done() -> bool:
            if not flight_over():
                return False
            if attack_type == "none" or not attack_started:
                return True
            return (attack_end_ts is not None
                    and time.time() >= attack_end_ts + ATTACK_POST_WINDOW_S)

        def result(timed_out: bool) -> dict:
            return {
                "mission_complete": mission_complete,
                "timed_out": timed_out,
                "reached": reached,
                "attack_start": attack_start_ts,
                "attack_end": attack_end_ts,
                "disarm_time": disarm_time,
                "disarm_reason": disarm_reason,
                "mode_changes": mode_changes,
                "statustext": statustext,
                "landed_by_orchestrator": landed_by_orchestrator,
            }

        while time.time() < deadline and self.keep_running:
            msg = self.conn.recv_match(
                type=["STATUSTEXT", "MISSION_ITEM_REACHED", "HEARTBEAT",
                      "GLOBAL_POSITION_INT"],
                blocking=True, timeout=2.0
            )

            # Check if attack conditions are met
            if (not attack_ready and attack_type != "none"
                    and first_wp_time is not None
                    and time.time() > min_flight_time
                    and time.time() - first_wp_time > 10):
                attack_ready = True
                log.info(f"  Attack window open (first WP + 60s flying)")

            # Start attack at randomized offset (only while still flying)
            if (attack_ready and not attack_started and not flight_over()
                    and attack_start_offset is not None
                    and time.time() - first_wp_time >= attack_start_offset):
                attack_proc = self.start_attack(attack_type, mission_id,
                                                attack_variant)
                attack_start_ts = time.time()
                attack_started = True
                log.info(f"  [!] Attack {attack_type} STARTED at offset "
                         f"{time.time() - first_wp_time:.0f}s")

            # Stop attack after its scheduled duration
            if (attack_started and attack_proc is not None
                    and attack_end_ts is None
                    and time.time() - attack_start_ts >= attack_duration):
                stop_process(attack_proc, f"attack {attack_type}")
                attack_end_ts = time.time()
                log.info(f"  [!] Attack {attack_type} STOPPED after "
                         f"{attack_duration:.0f}s")

            if msg is not None:
                mtype = msg.get_type()
                now = time.time()

                if mtype == "MISSION_ITEM_REACHED":
                    reached.add(msg.seq)
                    if first_wp_time is None:
                        first_wp_time = now
                    if last_seq > 0 and msg.seq == last_seq:
                        mission_complete = True
                    log.info(f"  WP {msg.seq} reached ({len(reached)} total)")

                elif mtype == "STATUSTEXT":
                    text = msg.text.strip()
                    log.info(f"  STATUSTEXT: {text}")
                    statustext.append([now, text])
                    if "Mission Complete" in text:
                        mission_complete = True
                    if ("Auto disarmed" in text or "Disarming" in text) \
                            and disarm_time is None:
                        disarm_time = now
                    # Keep the most specific disarm cause, e.g.
                    # "Crash: Disarming: AngErr=..." over "Disarming motors".
                    if ("Disarm" in text or "Crash" in text
                            or "Terminat" in text) and (
                            disarm_reason is None
                            or disarm_reason == "Disarming motors"):
                        disarm_reason = text

                elif mtype == "GLOBAL_POSITION_INT":
                    if msg.get_srcSystem() == 0:
                        pass
                    elif msg.relative_alt / 1000.0 > AIRBORNE_ALT_M:
                        was_airborne = True
                        on_ground_since = None
                    elif (was_airborne and disarm_time is None
                          and msg.relative_alt / 1000.0 < LANDED_ALT_M
                          and math.hypot(msg.vx, msg.vy) / 100.0 < LANDED_VH_MS
                          and abs(msg.vz) / 100.0 < LANDED_VZ_MS):
                        if on_ground_since is None:
                            on_ground_since = now
                        elif now - on_ground_since >= LANDED_HOLD_S:
                            log.info("  Landed but still armed — force disarm "
                                     "(orchestrator land detection)")
                            self.conn.mav.command_long_send(
                                self.conn.target_system,
                                self.conn.target_component,
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                0, 0, 21196, 0, 0, 0, 0, 0
                            )
                            landed_by_orchestrator = True
                            disarm_time = now
                            disarm_reason = "landed, force-disarmed by orchestrator"
                            # Landed after the last navigation item: the
                            # mission was flown completely (RTL landing).
                            if last_seq > 0 and reached and max(reached) >= last_seq - 1:
                                mission_complete = True
                    else:
                        on_ground_since = None

                elif mtype == "HEARTBEAT":
                    if (msg.type == mavutil.mavlink.MAV_TYPE_QUADROTOR
                            and msg.get_srcSystem() != 0):
                        new_mode = mavutil.mode_string_v10(msg)
                        if new_mode != mode:
                            mode = new_mode
                            mode_changes.append([now, mode])
                        armed = (msg.base_mode
                                 & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        if (not armed and disarm_time is None
                                and len(reached) > 0
                                and now > min_flight_time):
                            disarm_time = now
                            log.info(f"  Vehicle disarmed. "
                                     f"WPs reached: {len(reached)}")

            if run_done():
                log.info(f"  Run done (mission_complete={mission_complete}, "
                         f"disarmed={disarm_time is not None}). "
                         f"WPs reached: {len(reached)}")
                return result(timed_out=False)

        # Timeout (or shutdown) — clean up attack
        if attack_proc is not None and attack_end_ts is None:
            stop_process(attack_proc, f"attack {attack_type}")
            attack_end_ts = time.time()

        log.warning(f"  Timeout. Waypoints reached: {len(reached)}")
        return result(timed_out=True)

    def write_label(self, run_id: str, start_time: float, end_time: float,
                    attack_type: str = "none", attack_start: float = None,
                    attack_end: float = None, waypoints_reached: list = None,
                    mission_success: bool = False, attack_variant: str = None,
                    flight: dict = None, wind: dict = None, notes: str = ""):
        flight = flight or {}
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
            "attack_variant": attack_variant,
            "timed_out": flight.get("timed_out"),
            "disarm_time": flight.get("disarm_time"),
            "disarm_reason": flight.get("disarm_reason"),
            "mode_changes": flight.get("mode_changes", []),
            "statustext": flight.get("statustext", []),
            "landed_by_orchestrator": flight.get("landed_by_orchestrator", False),
            "wind": wind,
            "notes": notes
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

    def reset_vehicle_pose(self) -> bool:
        """Put the Gazebo model back to its upright spawn pose.

        A SITL reboot resets ArduPilot but not the Gazebo world: after a
        crash the model stays upside down (no takeoff possible), and after
        a landing away from home (failsafe, attack) the next mission would
        start there instead of at the spawn point.
        """
        try:
            r = subprocess.run(
                ["gz", "service", "-s", f"/world/{GZ_WORLD}/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "3000",
                 "--req", f'name: "{GZ_MODEL}", {GZ_SPAWN_POSE}'],
                timeout=10, capture_output=True, text=True
            )
            ok = "data: true" in r.stdout
        except Exception as e:
            log.warning(f"  Pose reset failed ({e})")
            return False
        if ok:
            log.info("[+] Vehicle reset to spawn pose")
        else:
            log.warning(f"  Pose reset failed: {r.stdout.strip()} {r.stderr.strip()}")
        return ok

    def reboot_sitl(self):
        """Reset vehicle pose and SITL state between missions.

        Force-disarm first: a vehicle still hovering after a timeout
        (failsafe hold) would otherwise refuse the reboot, and the pose
        reset must not teleport a powered vehicle.
        """
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0  # param2=21196 = force disarm
        )
        time.sleep(2)
        self.reset_vehicle_pose()
        time.sleep(1)
        log.info("[+] Rebooting SITL...")
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        time.sleep(20)
        self.conn = mavutil.mavlink_connection(self.connection)
        if not self._bind_vehicle_heartbeat():
            log.warning("[?] Only phantom heartbeat after reboot — retrying bind")
            if not self._bind_vehicle_heartbeat():
                # No vehicle on the link (radio/proxy chain down). Continuing
                # would bind to MAVProxy's phantom sysid 0 and send every
                # command into the void. Stop and let the watchdog restart
                # the stack.
                log.error("No vehicle heartbeat after reboot — stopping.")
                notify("[UAV-IDS] No vehicle heartbeat after reboot — stopping.")
                self.keep_running = False
                (self.output_dir / ".stop_orchestrator").touch()
                return
        log.info(f"[+] SITL rebooted, heartbeat OK "
                 f"(sys {self.conn.target_system}:{self.conn.target_component}).")

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

    def archive_crashed_run(self, mission_id: str, seed: int):
        """Move pcap, telemetry, wind series and label of a crashed run to
        <output>/_crashed_runs/<mission_id>_seed<seed>/ so the re-flight
        does not mix with it."""
        dest = self.output_dir / CRASHED_RUNS_DIR / f"{mission_id}_seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        files = [self.output_dir / "pcap" / f"{mission_id}.pcap",
                 self.output_dir / "wind" / f"{mission_id}_wind.csv",
                 self.output_dir / "phase_labels" / f"{mission_id}.json"]
        files += sorted((self.output_dir / "telemetry").glob(f"{mission_id}_*.csv"))
        for p in files:
            if p.exists():
                p.rename(dest / p.name)
        log.info(f"  Crashed run archived -> {dest}")

    def run_single(self, mission: dict, crashed_seeds: list = None):
        mission_id = mission["mission_id"]
        crashed_seeds = crashed_seeds or []
        wind_seed = int(mission_id.split("_")[-1]) + WIND_RESEED_OFFSET * len(crashed_seeds)
        log.info(f"=== Starting {mission_id} ({mission['profile']}) ===")
        wp_path = self.output_dir / "waypoints" / mission["waypoint_file"]

        # Load full params from meta JSON
        meta_path = self.output_dir / "meta" / f"{mission_id}_params.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Start capture
        tcpdump_proc = self.start_tcpdump(mission_id)
        telemetry_proc = self.start_telemetry_logger(mission_id)
        time.sleep(1)

        start_time = time.time()
        success = False
        airborne = False
        reached_wps = set()

        atk_start = None
        atk_end = None
        flight = {}
        wind = None
        attack_variant = meta.get("attack_variant")

        try:
            self.zero_wind()
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
            airborne = True

            # Apply wind only after a successful takeoff. Applying it before
            # (at mission start) made the vehicle fight the wind during the
            # vertical GUIDED climb, which the more aggressive drone presets
            # could not complete under moderate/gusty wind -> "Takeoff failed".
            # Wind is still active for the whole waypoint mission and the
            # attack window (attacks fire mid-flight), only the ~12 s takeoff
            # is calm.
            wind = self.start_wind(mission_id, meta["wind_params"],
                                   meta["wind_direction_deg"], seed=wind_seed)
            if crashed_seeds:
                wind["crashed_seeds"] = crashed_seeds

            # Now switch to AUTO — mission continues from WP2
            self.set_mode("AUTO")

            # Attack scheduling
            attack_type = meta.get("attack_type", "none")
            attack_start_offset = None
            attack_duration = None

            if attack_type != "none":
                # Deterministic schedule from per-mission attack_seed (meta):
                # start 60-180s after first WP, duration 15-60s. Falls back to
                # the global RNG if no seed is present. Seed may be an int or a
                # hex string.
                seed = meta.get("attack_seed")
                if isinstance(seed, str):
                    seed = int(seed, 16)
                rng = random.Random(seed) if seed is not None else random
                # Offset/duration kept short so the attack reliably fires within
                # even the shortest missions (a rectangle flies ~120s after the
                # first WP); offset+duration <= 85s leaves a normal window before
                # (min 60s flight is enforced) and after the attack.
                attack_start_offset = rng.uniform(15, 45)
                attack_duration = rng.uniform(15, 40)
                log.info(f"  Attack scheduled: {attack_type}, "
                         f"offset={attack_start_offset:.0f}s, "
                         f"duration={attack_duration:.0f}s "
                         f"(seed={seed})")

            mission_timeout = self.estimate_mission_timeout(wp_path, meta)
            flight = self.wait_mission_complete(
                timeout=mission_timeout,
                attack_type=attack_type,
                attack_start_offset=attack_start_offset,
                attack_duration=attack_duration,
                mission_id=mission_id,
                attack_variant=attack_variant
            )
            success = flight["mission_complete"]
            reached_wps = flight["reached"]
            atk_start = flight["attack_start"]
            atk_end = flight["attack_end"]

            if not success:
                log.warning(f"  Mission {mission_id} did not complete cleanly")
        except Exception as e:
            log.error(f"Run {mission_id} failed: {e}")
        finally:
            end_time = time.time()
            try:
                self.stop_wind()
            except Exception as e:
                log.error(f"  Could not reset wind: {e}")
            stop_process(tcpdump_proc, "tcpdump")
            stop_process(telemetry_proc, "telemetry logger")
            notes = ""
            if crashed_seeds:
                notes = (f"re-flown with wind seed {wind_seed} after crash "
                         f"with seed(s) {crashed_seeds}")
            self.write_label(mission_id, start_time, end_time,
                             attack_type=meta.get("attack_type", "none"),
                             attack_start=atk_start,
                             attack_end=atk_end,
                             waypoints_reached=sorted(reached_wps),
                             mission_success=success,
                             attack_variant=attack_variant,
                             flight=flight, wind=wind, notes=notes)
            # For attack runs a flight that got airborne and actually fired the
            # attack is valid data even if it never completed cleanly — the
            # failsafe/hold/termination is the attack effect, not a sim failure.
            # Only genuine pre-flight errors (arming/takeoff/GPS) count as
            # failures and toward the consecutive-failure stop.
            run_attack_type = meta.get("attack_type", "none")
            if run_attack_type != "none":
                run_valid = airborne and atk_start is not None
            else:
                run_valid = success

            crashed = str(flight.get("disarm_reason") or "").startswith("Crash")
            reflight = (run_attack_type == "none" and crashed
                        and not crashed_seeds and self.keep_running)
            if reflight:
                # Neither completed nor failed: archive and fly again below
                log.warning(f"  {mission_id} crashed with wind seed {wind_seed} "
                            f"— re-flying once with seed "
                            f"{wind_seed + WIND_RESEED_OFFSET}")
                self.archive_crashed_run(mission_id, wind_seed)
            else:
                status = "completed" if run_valid else "failed"
                self.update_manifest_status(mission_id, status)
                log.info(f"=== {mission_id} {status} ({end_time - start_time:.1f}s) ===")

                if run_valid:
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

            if self.keep_running:
                self.reboot_sitl()

        if reflight and self.keep_running:
            self.wait_while_paused()
            if self.keep_running:
                self.run_single(mission, crashed_seeds=crashed_seeds + [wind_seed])

    def wait_while_paused(self):
        """Hold between missions while the watchdog's pause flag exists,
        then reboot the SITL (gazebo/ardupilot may have been restarted)."""
        flag = self.output_dir / PAUSE_FLAG
        if not flag.exists():
            return
        log.info("Paused (watchdog flag) — waiting")
        while flag.exists() and self.keep_running:
            time.sleep(2)
        if self.keep_running:
            log.info("Resumed — rebooting SITL")
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
        # Start from a clean state (upright at the spawn point, fresh SITL),
        # whatever the previous run or a watchdog restart left behind.
        self.reboot_sitl()
        if not self.keep_running:
            return

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
            self.wait_while_paused()
            if not self.keep_running:
                continue
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