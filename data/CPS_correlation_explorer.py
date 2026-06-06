#!/usr/bin/env python3
"""
cps_correlation_explorer.py — Generate multiple CPS correlation plots
to find which sensor vs. network combinations show clear correlations.

Usage:
    python3 cps_correlation_explorer.py --dataset ./dataset --mission mission_00028 --save
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(filepath):
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
    for col in data:
        data[col] = np.array(data[col])
    return data


def ts_to_seconds(ts_array):
    return (ts_array - ts_array[0]) / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mission", type=str, required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    d = args.dataset
    m = args.mission

    # Load all data
    gpi = load_csv(d / "telemetry" / f"{m}_GLOBAL_POSITION_INT.csv")
    imu = load_csv(d / "telemetry" / f"{m}_RAW_IMU.csv")
    servo = load_csv(d / "telemetry" / f"{m}_SERVO_OUTPUT_RAW.csv")
    press = load_csv(d / "telemetry" / f"{m}_SCALED_PRESSURE.csv")
    net = load_csv(d / "network" / f"{m}_network.csv")

    # Time axes
    t_gpi = ts_to_seconds(gpi["host_timestamp_ns"])
    t_imu = ts_to_seconds(imu["host_timestamp_ns"])
    t_servo = ts_to_seconds(servo["host_timestamp_ns"])
    t_press = ts_to_seconds(press["host_timestamp_ns"])

    t0_net = net["timestamp_s"][0]
    t_net = net["timestamp_s"] - t0_net

    # Derived signals
    alt = gpi["relative_alt"] / 1000.0
    motor_avg = (servo["servo1_raw"] + servo["servo2_raw"] +
                 servo["servo3_raw"] + servo["servo4_raw"]) / 4.0
    motor_spread = (np.max([servo["servo1_raw"], servo["servo2_raw"],
                            servo["servo3_raw"], servo["servo4_raw"]], axis=0) -
                    np.min([servo["servo1_raw"], servo["servo2_raw"],
                            servo["servo3_raw"], servo["servo4_raw"]], axis=0))
    ground_speed = np.sqrt(gpi["vx"]**2 + gpi["vy"]**2) / 100.0  # cm/s to m/s
    vertical_speed = gpi["vz"] / 100.0  # cm/s to m/s
    yaw_rate = np.abs(imu["zgyro"])
    roll_rate = np.abs(imu["xgyro"])
    pressure = press["press_abs"]

    # Bin network data into 2s windows
    t_max = max(t_net[-1], 1)
    bin_edges = np.arange(0, t_max + 2, 2.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    frame_rate = np.histogram(t_net, bins=bin_edges)[0] / 2.0  # per second

    avg_payload = np.zeros(len(bin_centers))
    total_bytes = np.zeros(len(bin_centers))
    dir_mask_gcs = net["direction"] == 3
    for i in range(len(bin_centers)):
        mask = (t_net >= bin_edges[i]) & (t_net < bin_edges[i+1])
        if mask.sum() > 0:
            avg_payload[i] = net["payload_len"][mask].mean()
            total_bytes[i] = net["frame_len"][mask].sum()

    # Resample physical signals to bin_centers
    def resample(t_phys, signal, centers):
        out = np.zeros(len(centers))
        for i, tc in enumerate(centers):
            mask = (t_phys >= tc - 1) & (t_phys < tc + 1)
            if mask.sum() > 0:
                out[i] = signal[mask].mean()
        return out

    alt_r = resample(t_gpi, alt, bin_centers)
    motor_r = resample(t_servo, motor_avg, bin_centers)
    spread_r = resample(t_servo, motor_spread, bin_centers)
    speed_r = resample(t_gpi, ground_speed, bin_centers)
    vspeed_r = resample(t_gpi, vertical_speed, bin_centers)
    yaw_r = resample(t_imu, yaw_rate, bin_centers)
    roll_r = resample(t_imu, roll_rate, bin_centers)
    press_r = resample(t_press, pressure, bin_centers)

    # Physical signals to plot
    phys_signals = [
        ("Altitude (m)", alt_r, "#2196F3"),
        ("Motor Avg PWM", motor_r, "#FF9800"),
        ("Motor Spread PWM", spread_r, "#E91E63"),
        ("Ground Speed (m/s)", speed_r, "#4CAF50"),
        ("Vertical Speed (m/s)", vspeed_r, "#9C27B0"),
        ("|Yaw Rate| (raw)", yaw_r, "#F44336"),
        ("|Roll Rate| (raw)", roll_r, "#795548"),
        ("Pressure (hPa)", press_r, "#607D8B"),
    ]

    # Network signals to plot against
    net_signals = [
        ("Frame Rate (f/s)", frame_rate, "#00BCD4"),
        ("Avg Payload (bytes)", avg_payload, "#FF5722"),
        ("Throughput (bytes/s)", total_bytes / 2.0, "#3F51B5"),
    ]

    # Plot grid: 8 physical × 3 network = 24 subplots
    n_phys = len(phys_signals)
    n_net = len(net_signals)

    fig, axes = plt.subplots(n_phys, n_net, figsize=(6 * n_net, 3.5 * n_phys),
                              sharex=True)
    fig.suptitle(f"CPS Correlation Explorer — {m}", fontsize=16,
                 fontweight="bold", y=1.01)

    for row, (p_name, p_signal, p_color) in enumerate(phys_signals):
        for col, (n_name, n_signal, n_color) in enumerate(net_signals):
            ax = axes[row, col]
            ax2 = ax.twinx()

            l1, = ax.plot(bin_centers, p_signal, color=p_color, linewidth=1.2,
                    alpha=0.9, label=p_name)
            l2, = ax2.plot(bin_centers, n_signal, color=n_color, linewidth=1.2,
                     alpha=0.7, linestyle="--", label=n_name)

            # Column title (top row only)
            if row == 0:
                ax.set_title(n_name, fontsize=12, fontweight="bold", pad=10)

            # Y-axis labels
            ax.set_ylabel(p_name, fontsize=9, color=p_color, fontweight="bold")
            ax2.set_ylabel(n_name, fontsize=8, color=n_color)

            # X-axis label (bottom row only)
            if row == n_phys - 1:
                ax.set_xlabel("Time (s)", fontsize=10)

            # Legend in every subplot
            ax.legend([l1, l2], [p_name, n_name], fontsize=7,
                      loc="upper right", framealpha=0.8)

            ax.tick_params(axis="y", labelcolor=p_color, labelsize=8)
            ax2.tick_params(axis="y", labelcolor=n_color, labelsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if args.save:
        out = args.dataset / f"{m}_cps_explorer.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()