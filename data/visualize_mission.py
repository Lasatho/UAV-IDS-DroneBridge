#!/usr/bin/env python3
"""
visualize_mission.py — Plot all captured sensor + network data for a single mission.

Generates a multi-panel figure showing:
  Physical layer:
    1. 3D trajectory (Ist) vs planned waypoints (Soll)
    2. 2D ground track (lat/lon) with waypoints
    3. Altitude profile over time
    4. IMU accelerations (x, y, z)
    5. IMU gyroscope (x, y, z)
    6. Servo/motor outputs (channels 1-4)
    7. GPS quality (fix_type, satellites, eph)
    8. Barometric pressure
  Network layer:
    9.  Frame timeline (size over time, colored by direction)
    10. Payload size distribution (histogram)
    11. Sequence number progression per direction
    12. Traffic direction balance over time

Usage:
    python3 visualize_mission.py --dataset ./dataset --mission mission_00028
    python3 visualize_mission.py --dataset ./dataset --mission mission_00028 --save
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(filepath: Path) -> dict:
    """Load a CSV into a dict of column_name -> list of float values."""
    data = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            data[col] = []
        for row in reader:
            for col in reader.fieldnames:
                try:
                    data[col].append(float(row[col]))
                except (ValueError, TypeError):
                    data[col].append(0.0)
    # Convert to numpy arrays
    for col in data:
        data[col] = np.array(data[col])
    return data


def load_waypoints(wp_file: Path) -> List[Tuple[float, float, float]]:
    """Load waypoints as list of (lat, lon, alt) tuples."""
    wps = []
    with open(wp_file, "r") as f:
        header = f.readline()  # skip QGC WPL 110
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            cmd = int(parts[3])
            lat = float(parts[8])
            lon = float(parts[9])
            alt = float(parts[10])
            # Only navigable waypoints (skip RTL with 0,0 coords)
            if lat != 0.0 or lon != 0.0:
                wps.append((lat, lon, alt))
    return wps


def ts_to_seconds(ts_array: np.ndarray) -> np.ndarray:
    """Convert host_timestamp_ns to relative seconds from start."""
    t0 = ts_array[0]
    return (ts_array - t0) / 1e9


def latlon_to_meters(lat: np.ndarray, lon: np.ndarray,
                     ref_lat: float, ref_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon (degE7 or degrees) to meters relative to reference."""
    # Auto-detect degE7
    if np.abs(lat[0]) > 900:
        lat = lat / 1e7
        lon = lon / 1e7

    R = 6378137.0
    north = np.radians(lat - ref_lat) * R
    east = np.radians(lon - ref_lon) * R * np.cos(np.radians(ref_lat))
    return north, east


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mission(dataset_dir: Path, mission_id: str, save: bool = False):
    telem_dir = dataset_dir / "telemetry"
    wp_dir = dataset_dir / "waypoints"
    meta_dir = dataset_dir / "meta"
    label_dir = dataset_dir / "phase_labels"

    # Load all data
    gpi_file = telem_dir / f"{mission_id}_GLOBAL_POSITION_INT.csv"
    imu_file = telem_dir / f"{mission_id}_RAW_IMU.csv"
    servo_file = telem_dir / f"{mission_id}_SERVO_OUTPUT_RAW.csv"
    gps_file = telem_dir / f"{mission_id}_GPS_RAW_INT.csv"
    press_file = telem_dir / f"{mission_id}_SCALED_PRESSURE.csv"

    for f in [gpi_file, imu_file, servo_file, gps_file, press_file]:
        if not f.exists():
            print(f"Missing: {f}")
            sys.exit(1)

    gpi = load_csv(gpi_file)
    imu = load_csv(imu_file)
    servo = load_csv(servo_file)
    gps = load_csv(gps_file)
    press = load_csv(press_file)

    # Load network data (optional — skip network panels if missing)
    network_dir = dataset_dir / "network"
    net_file = network_dir / f"{mission_id}_network.csv"
    net = None
    if net_file.exists():
        net = load_csv(net_file)
    else:
        print(f"Note: No network data at {net_file} — skipping network panels")

    # Load waypoints
    wp_file = wp_dir / f"{mission_id}.waypoints"
    waypoints = load_waypoints(wp_file)

    # Load meta
    meta_file = meta_dir / f"{mission_id}_params.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)

    # Load label
    label_file = label_dir / f"{mission_id}.json"
    label = {}
    if label_file.exists():
        with open(label_file, "r") as f:
            label = json.load(f)

    # Time axes (relative seconds)
    t_gpi = ts_to_seconds(gpi["host_timestamp_ns"])
    t_imu = ts_to_seconds(imu["host_timestamp_ns"])
    t_servo = ts_to_seconds(servo["host_timestamp_ns"])
    t_gps = ts_to_seconds(gps["host_timestamp_ns"])
    t_press = ts_to_seconds(press["host_timestamp_ns"])

    # Position in meters (relative to home)
    ref_lat = meta["home"]["lat"]
    ref_lon = meta["home"]["lng"]

    north, east = latlon_to_meters(gpi["lat"], gpi["lon"], ref_lat, ref_lon)
    alt_rel = gpi["relative_alt"] / 1000.0  # mm to m

    # Waypoints in meters
    wp_north = []
    wp_east = []
    wp_alt = []
    for wlat, wlon, walt in waypoints:
        wn = (wlat - ref_lat) * 6378137.0 * np.pi / 180.0
        we = (wlon - ref_lon) * 6378137.0 * np.pi / 180.0 * np.cos(np.radians(ref_lat))
        wp_north.append(wn)
        wp_east.append(we)
        wp_alt.append(walt)

    # Build subtitle
    profile = meta.get("profile", "?")
    preset = meta.get("drone_preset", "?")
    wind = meta.get("wind_profile", "?")
    duration = label.get("duration_s", 0)
    success = label.get("mission_success", False)
    status_str = "COMPLETED" if success else "FAILED"

    subtitle = (f"Profile: {profile} | Preset: {preset} | Wind: {wind} | "
                f"Duration: {duration:.0f}s | {status_str}")

    # --------------- FIGURE ---------------
    n_rows = 6 if net is not None else 4
    fig = plt.figure(figsize=(20, n_rows * 6))
    fig.suptitle(f"{mission_id}", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.98, subtitle, ha="center", fontsize=11, color="gray")

    # Color scheme
    c_traj = "#2196F3"
    c_wp = "#F44336"
    c_start = "#4CAF50"
    c_end = "#FF9800"
    c_to_drone = "#E91E63"
    c_to_gcs = "#00BCD4"

    # --- 1. 3D Trajectory ---
    ax1 = fig.add_subplot(n_rows, 2, 1, projection="3d")
    ax1.plot(east, north, alt_rel, color=c_traj, linewidth=0.8, alpha=0.8)
    ax1.scatter(wp_east, wp_north, wp_alt, c=c_wp, s=80, marker="^",
                label="Waypoints", zorder=5, edgecolors="white", linewidths=0.5)
    if len(east) > 0:
        ax1.scatter([east[0]], [north[0]], [alt_rel[0]], c=c_start, s=100,
                    marker="o", label="Start", zorder=6)
        ax1.scatter([east[-1]], [north[-1]], [alt_rel[-1]], c=c_end, s=100,
                    marker="s", label="End", zorder=6)
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_zlabel("Alt (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend(fontsize=8)

    # --- 2. 2D Ground Track ---
    ax2 = fig.add_subplot(n_rows, 2, 2)
    ax2.plot(east, north, color=c_traj, linewidth=0.8, alpha=0.8)
    ax2.scatter(wp_east, wp_north, c=c_wp, s=80, marker="^",
                label="Waypoints", zorder=5, edgecolors="white", linewidths=0.5)
    for i, (we, wn) in enumerate(zip(wp_east, wp_north)):
        ax2.annotate(f"WP{i}", (we, wn), fontsize=7, ha="left",
                     xytext=(5, 5), textcoords="offset points")
    if len(east) > 0:
        ax2.scatter([east[0]], [north[0]], c=c_start, s=100, marker="o",
                    label="Start", zorder=6)
        ax2.scatter([east[-1]], [north[-1]], c=c_end, s=100, marker="s",
                    label="End", zorder=6)
    ax2.set_xlabel("East (m)")
    ax2.set_ylabel("North (m)")
    ax2.set_title("Ground Track (2D)")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # --- 3. Altitude Profile ---
    ax3 = fig.add_subplot(n_rows, 2, 3)
    ax3.plot(t_gpi, alt_rel, color=c_traj, linewidth=0.8)
    ax3.axhline(y=0, color="brown", linestyle="--", alpha=0.5, label="Ground")
    for walt in set(wp_alt):
        ax3.axhline(y=walt, color=c_wp, linestyle=":", alpha=0.3)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Altitude (m)")
    ax3.set_title("Altitude Profile")
    ax3.grid(True, alpha=0.3)

    # --- 4. IMU Accelerations ---
    ax4 = fig.add_subplot(n_rows, 2, 4)
    ax4.plot(t_imu, imu["xacc"], label="X", linewidth=0.5, alpha=0.8)
    ax4.plot(t_imu, imu["yacc"], label="Y", linewidth=0.5, alpha=0.8)
    ax4.plot(t_imu, imu["zacc"], label="Z", linewidth=0.5, alpha=0.8)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Acceleration (raw)")
    ax4.set_title("IMU Accelerations")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- 5. IMU Gyroscope ---
    ax5 = fig.add_subplot(n_rows, 2, 5)
    ax5.plot(t_imu, imu["xgyro"], label="X", linewidth=0.5, alpha=0.8)
    ax5.plot(t_imu, imu["ygyro"], label="Y", linewidth=0.5, alpha=0.8)
    ax5.plot(t_imu, imu["zgyro"], label="Z", linewidth=0.5, alpha=0.8)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Angular rate (raw)")
    ax5.set_title("IMU Gyroscope")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- 6. Servo/Motor Outputs ---
    ax6 = fig.add_subplot(n_rows, 2, 6)
    ax6.plot(t_servo, servo["servo1_raw"], label="M1", linewidth=0.5, alpha=0.8)
    ax6.plot(t_servo, servo["servo2_raw"], label="M2", linewidth=0.5, alpha=0.8)
    ax6.plot(t_servo, servo["servo3_raw"], label="M3", linewidth=0.5, alpha=0.8)
    ax6.plot(t_servo, servo["servo4_raw"], label="M4", linewidth=0.5, alpha=0.8)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("PWM (µs)")
    ax6.set_title("Motor Outputs")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # --- 7. GPS Quality ---
    ax7 = fig.add_subplot(n_rows, 2, 7)
    ax7_twin = ax7.twinx()
    ax7.plot(t_gps, gps["satellites_visible"], color="#4CAF50",
             linewidth=0.8, label="Satellites")
    ax7_twin.plot(t_gps, gps["eph"], color="#FF5722",
                  linewidth=0.8, alpha=0.7, label="EPH")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Satellites", color="#4CAF50")
    ax7_twin.set_ylabel("EPH (cm)", color="#FF5722")
    ax7.set_title("GPS Quality")
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax7.grid(True, alpha=0.3)

    # --- 8. Barometric Pressure ---
    ax8 = fig.add_subplot(n_rows, 2, 8)
    ax8.plot(t_press, press["press_abs"], color="#9C27B0", linewidth=0.8)
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Pressure (hPa)")
    ax8.set_title("Barometric Pressure")
    ax8.grid(True, alpha=0.3)

    # --------------- NETWORK PANELS ---------------
    if net is not None:
        # Network time axis (relative seconds from first frame)
        t0_net = net["timestamp_s"][0]
        t_net = net["timestamp_s"] - t0_net

        dir_mask_drone = net["direction"] == 1   # to drone
        dir_mask_gcs = net["direction"] == 3     # to GCS

        # Bin into 2s windows for frame rate
        t_max = max(t_net[-1], t_gpi[-1]) if len(t_gpi) > 0 else t_net[-1]
        bin_edges = np.arange(0, t_max + 2, 2.0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fr_to_drone = np.histogram(t_net[dir_mask_drone], bins=bin_edges)[0] / 2.0
        fr_to_gcs = np.histogram(t_net[dir_mask_gcs], bins=bin_edges)[0] / 2.0

        # --- 9. Frame Rate over Time ---
        ax9 = fig.add_subplot(n_rows, 2, 9)
        ax9.bar(bin_centers - 0.4, fr_to_gcs, width=1.6, alpha=0.7,
                color=c_to_gcs, label="Drone → GCS")
        ax9.bar(bin_centers + 0.4, fr_to_drone, width=1.6, alpha=0.7,
                color=c_to_drone, label="GCS → Drone")
        ax9.set_xlabel("Time (s)")
        ax9.set_ylabel("Frames / s")
        ax9.set_title("DroneBridge Frame Rate")
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)

        # --- 10. Payload Size Distribution ---
        ax10 = fig.add_subplot(n_rows, 2, 10)
        bins = np.linspace(0, max(net["payload_len"].max(), 1), 40)
        ax10.hist(net["payload_len"][dir_mask_drone], bins=bins, alpha=0.7,
                  color=c_to_drone, label="GCS → Drone")
        ax10.hist(net["payload_len"][dir_mask_gcs], bins=bins, alpha=0.7,
                  color=c_to_gcs, label="Drone → GCS")
        ax10.set_xlabel("Payload Length (bytes)")
        ax10.set_ylabel("Frame Count")
        ax10.set_title("Payload Size Distribution")
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)

        # --- 11. Summary Table ---
        ax11 = fig.add_subplot(n_rows, 2, (11, 12))
        ax11.axis("off")

        n_total = len(t_net)
        n_to_drone = int(dir_mask_drone.sum())
        n_to_gcs = int(dir_mask_gcs.sum())
        bytes_to_drone = net["frame_len"][dir_mask_drone].sum()
        bytes_to_gcs = net["frame_len"][dir_mask_gcs].sum()
        avg_payload_drone = (net["payload_len"][dir_mask_drone].mean()
                             if n_to_drone > 0 else 0)
        avg_payload_gcs = (net["payload_len"][dir_mask_gcs].mean()
                           if n_to_gcs > 0 else 0)
        mission_dur = t_net[-1] - t_net[0] if len(t_net) > 1 else 0
        avg_fps = n_total / max(mission_dur, 1)

        table_data = [
            ["Total Frames", f"{n_total}"],
            ["Drone → GCS", f"{n_to_gcs} frames ({bytes_to_gcs/1024:.1f} KB)"],
            ["GCS → Drone", f"{n_to_drone} frames ({bytes_to_drone/1024:.1f} KB)"],
            ["Avg Payload Drone→GCS", f"{avg_payload_gcs:.0f} bytes"],
            ["Avg Payload GCS→Drone", f"{avg_payload_drone:.0f} bytes"],
            ["Avg Frame Rate", f"{avg_fps:.1f} frames/s"],
            ["DL/UL Ratio", f"{bytes_to_gcs/max(bytes_to_drone,1):.1f}:1"],
            ["Comm ID / Port", f"{int(net['comm_id'][0])} / {int(net['port'][0])}"],
            ["Duration", f"{mission_dur:.1f}s"],
        ]

        table = ax11.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            colColours=["#E3F2FD", "#E3F2FD"],
            loc="center",
            colWidths=[0.35, 0.35]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor("#BBDEFB")
            if key[0] == 0:
                cell.set_facecolor("#1565C0")
                cell.set_text_props(color="white", fontweight="bold")
        ax11.set_title("Network Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        out_path = dataset_dir / f"{mission_id}_visualization.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all sensor data for a single mission"
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset root directory")
    parser.add_argument("--mission", type=str, required=True,
                        help="Mission ID (e.g. mission_00028)")
    parser.add_argument("--save", action="store_true",
                        help="Save to PNG instead of showing")
    args = parser.parse_args()

    plot_mission(args.dataset, args.mission, args.save)


if __name__ == "__main__":
    main()