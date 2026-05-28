#!/usr/bin/env python3
"""
pcap_to_csv.py — Extract raw DroneBridge frame fields from PCAP files.

Reads hwsim0 captures and extracts per-frame fields from the
DroneBridge Raw Protocol v2 header at fixed byte offsets.

Frame layout:
  Bytes  0-21: Radiotap Header (22 bytes, fixed)
  Bytes 22-25: FCF + Duration (0x08 0x00 0x00 0x00 for Data)
  Byte  26:    Direction (0x01 = to drone, 0x03 = to GCS)
  Byte  27:    Comm ID
  Byte  28:    Port (1=Control, 2=Telemetry, 3=Video, 4=Comms, 5=Status, 6=Proxy, 7=RC)
  Bytes 29-30: Payload Length (uint16 LE)
  Byte  31:    Sequence Number (0-255)
  Bytes 32+:   Payload (AES+EAX encrypted)

Output CSV columns:
  timestamp_s, frame_len, direction, comm_id, port, payload_len, seq_num

Usage:
    python3 pcap_to_csv.py --pcap ./dataset/pcap/mission_00028.pcap
    python3 pcap_to_csv.py --pcap ./dataset/pcap/mission_00028.pcap --output ./output.csv
    python3 pcap_to_csv.py --dataset ./dataset                      # batch all pcaps
"""

import argparse
import csv
import struct
import sys
from pathlib import Path


# DroneBridge header field offsets (from frame start, after 22-byte Radiotap)
RADIOTAP_LEN = 22
OFFSET_DIRECTION = 26
OFFSET_COMM_ID = 27
OFFSET_PORT = 28
OFFSET_PAYLOAD_LEN = 29  # 2 bytes, uint16 LE
OFFSET_SEQ_NUM = 31

# Minimum frame size: 22 (radiotap) + 10 (DB header) = 32 bytes
MIN_FRAME_SIZE = 32


def parse_pcap_frames(pcap_path: Path):
    """
    Parse PCAP file manually (libpcap format) to avoid external dependencies.
    Yields (timestamp_s, raw_frame_bytes) for each packet.
    """
    with open(pcap_path, "rb") as f:
        # Global header: 24 bytes
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError(f"File too short: {pcap_path}")

        magic = struct.unpack("<I", global_header[0:4])[0]
        if magic == 0xa1b2c3d4:
            # Little-endian, microsecond timestamps
            endian = "<"
            ts_resolution = 1e-6
        elif magic == 0xa1b23c4d:
            # Little-endian, nanosecond timestamps
            endian = "<"
            ts_resolution = 1e-9
        elif magic == 0xd4c3b2a1:
            # Big-endian, microsecond timestamps
            endian = ">"
            ts_resolution = 1e-6
        elif magic == 0x4d3cb2a1:
            # Big-endian, nanosecond timestamps
            endian = ">"
            ts_resolution = 1e-9
        else:
            raise ValueError(f"Not a valid PCAP file: {pcap_path} "
                             f"(magic=0x{magic:08x})")

        while True:
            # Packet header: 16 bytes
            pkt_header = f.read(16)
            if len(pkt_header) < 16:
                break  # EOF

            ts_sec, ts_frac, incl_len, orig_len = struct.unpack(
                f"{endian}IIII", pkt_header
            )
            timestamp_s = ts_sec + ts_frac * ts_resolution

            # Packet data
            data = f.read(incl_len)
            if len(data) < incl_len:
                break  # truncated

            yield timestamp_s, data, orig_len


def extract_db_fields(frame_bytes: bytes, orig_len: int):
    """Extract DroneBridge header fields from raw frame bytes."""
    if len(frame_bytes) < MIN_FRAME_SIZE:
        return None

    direction = frame_bytes[OFFSET_DIRECTION]
    comm_id = frame_bytes[OFFSET_COMM_ID]
    port = frame_bytes[OFFSET_PORT]
    payload_len = struct.unpack_from("<H", frame_bytes, OFFSET_PAYLOAD_LEN)[0]
    seq_num = frame_bytes[OFFSET_SEQ_NUM]

    return {
        "frame_len": orig_len,
        "direction": direction,
        "comm_id": comm_id,
        "port": port,
        "payload_len": payload_len,
        "seq_num": seq_num,
    }


def convert_pcap(pcap_path: Path, output_path: Path):
    """Convert a single PCAP to CSV."""
    fieldnames = [
        "timestamp_s", "frame_len", "direction",
        "comm_id", "port", "payload_len", "seq_num"
    ]

    n_frames = 0
    n_skipped = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for timestamp_s, frame_bytes, orig_len in parse_pcap_frames(pcap_path):
            fields = extract_db_fields(frame_bytes, orig_len)
            if fields is None:
                n_skipped += 1
                continue

            fields["timestamp_s"] = f"{timestamp_s:.9f}"
            writer.writerow(fields)
            n_frames += 1

    return n_frames, n_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract DroneBridge frame fields from PCAP to CSV"
    )
    parser.add_argument("--pcap", type=Path, default=None,
                        help="Single PCAP file to convert")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: <pcap_name>.csv)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset root — batch convert all pcaps in dataset/pcap/")
    args = parser.parse_args()

    if args.pcap:
        # Single file mode
        output = args.output or args.pcap.with_suffix(".csv")
        print(f"Converting {args.pcap.name}...")
        n_frames, n_skipped = convert_pcap(args.pcap, output)
        print(f"  {n_frames} frames extracted, {n_skipped} skipped")
        print(f"  Output: {output}")

    elif args.dataset:
        # Batch mode
        pcap_dir = args.dataset / "pcap"
        network_dir = args.dataset / "network"
        network_dir.mkdir(exist_ok=True)

        pcaps = sorted(pcap_dir.glob("*.pcap"))
        if not pcaps:
            print(f"No pcap files in {pcap_dir}")
            sys.exit(1)

        print(f"Batch converting {len(pcaps)} PCAPs...")
        total_frames = 0
        total_skipped = 0

        for i, pcap_path in enumerate(pcaps):
            mission_id = pcap_path.stem
            output_path = network_dir / f"{mission_id}_network.csv"

            n_frames, n_skipped = convert_pcap(pcap_path, output_path)
            total_frames += n_frames
            total_skipped += n_skipped

            if (i + 1) % 100 == 0 or (i + 1) == len(pcaps):
                print(f"  [{i+1}/{len(pcaps)}] {total_frames} frames total")

        print(f"\nDone: {total_frames} frames, {total_skipped} skipped")
        print(f"Output: {network_dir}/")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()