#!/usr/bin/env python3
"""
pcap_to_csv.py — Extract raw DroneBridge frame fields from PCAP files.

Reads hwsim0 captures (pcap/, whole medium) or captures of the vehicle's
own interface (pcap_onboard/, wlan0) and extracts per-frame fields from the
DroneBridge Raw Protocol v2 header. Offsets are relative to the end of the
radiotap header, whose length is read from the header itself: 22 bytes on
hwsim0, variable on wlan0 (received frames carry the signal field, own
transmitted frames do not).

Frame layout (R = radiotap length, 22 on hwsim0):
  Bytes  0-R-1: Radiotap Header
  Bytes R..R+3: FCF + Duration (0x08 0x00 0x00 0x00 for Data)
  Byte  R+4:    Direction (0x01 = to drone, 0x03 = to GCS)
  Byte  R+5:    Comm ID
  Byte  R+6:    Port (1=Control, 2=Telemetry, 3=Video, 4=Comms, 5=Status, 6=Proxy, 7=RC)
  Bytes R+7..8: Payload Length (uint16 LE)
  Byte  R+9:    Sequence Number (0-255)
  Bytes R+10+:  Payload (cleartext MAVLink on port 1; the companion computer
               variant has no transport encryption)

Output CSV columns:
  host_timestamp_ns, timestamp_s, frame_len, direction, comm_id, port,
  payload_len, seq_num [, signal_dbm with --with-signal: radiotap
  dBm_AntSignal of received frames, empty for own transmitted frames]

A monitor capture of the vehicle's own interface records every frame the
interface transmits twice, byte-identical, the copy 0.4-8 ms later; the
medium (hwsim0) carries it once (checked 2026-11-03 on HP, 30 s of flight:
153 of 153 own frames paired). With --with-signal the copy is dropped.

timestamp_s is the capture time (epoch, tcpdump). host_timestamp_ns is the
same instant on the clock of the telemetry CSVs (time.monotonic_ns() of
the capture host), so the training pipeline can align the frames like a
telemetry message type. The two clocks differ by a constant per host and
boot (dataset/clock_offset_<host>_<date>.json, epoch_minus_monotonic_ns),
so convert on the host that recorded the pcap, before it reboots.

Usage:
    python3 pcap_to_csv.py --pcap ./dataset/pcap/mission_00028.pcap
    python3 pcap_to_csv.py --pcap ./dataset/pcap/mission_00028.pcap --output ./output.csv
    python3 pcap_to_csv.py --dataset ./dataset                      # batch all pcaps
    python3 pcap_to_csv.py --dataset ./dataset \
        --clock-offset ./dataset/clock_offset_HP-BURNER_20261025.json \
        --manifest ./dataset/missions_manifest_v2_merged.csv --host hp
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path


# DroneBridge header field offsets, relative to the end of the radiotap header
OFFSET_DIRECTION = 4
OFFSET_COMM_ID = 5
OFFSET_PORT = 6
OFFSET_PAYLOAD_LEN = 7  # 2 bytes, uint16 LE
OFFSET_SEQ_NUM = 9
DB_HEADER_LEN = 10
# Frames shorter than this are skipped: 22-byte hwsim0 radiotap + DroneBridge
# header. This drops the 24-byte frames of the rf_jamming jammer, which carry
# no DroneBridge header (decision 2026-11-02, variant a: keep them out).
MIN_FRAME_SIZE = 32

# Own transmitted frame repeated within this time = capture echo, dropped
TX_ECHO_WINDOW_S = 0.05

# Radiotap fields up to dBm_AntSignal (bit 5): (bit, alignment, size)
_RT_FIELDS = [(0, 8, 8), (1, 1, 1), (2, 1, 1), (3, 2, 4), (4, 2, 2), (5, 1, 1)]


def radiotap_signal_dbm(frame_bytes: bytes):
    """dBm_AntSignal of the radiotap header, None if the field is absent."""
    rt_len = struct.unpack_from("<H", frame_bytes, 2)[0]
    present = struct.unpack_from("<I", frame_bytes, 4)[0]
    if not present & (1 << 5):
        return None
    pos = 8
    word = present
    while word & (1 << 31):  # extended present bitmaps
        word = struct.unpack_from("<I", frame_bytes, pos)[0]
        pos += 4
    for bit, align, size in _RT_FIELDS:
        if not present & (1 << bit):
            continue
        pos = (pos + align - 1) // align * align
        if bit == 5:
            return struct.unpack_from("<b", frame_bytes, pos)[0] if pos < rt_len else None
        pos += size
    return None


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


def extract_db_fields(frame_bytes: bytes, orig_len: int, with_signal: bool = False):
    """Extract DroneBridge header fields from raw frame bytes."""
    if len(frame_bytes) < MIN_FRAME_SIZE:
        return None
    r = struct.unpack_from("<H", frame_bytes, 2)[0]
    if len(frame_bytes) < r + DB_HEADER_LEN:
        return None

    direction = frame_bytes[r + OFFSET_DIRECTION]
    comm_id = frame_bytes[r + OFFSET_COMM_ID]
    port = frame_bytes[r + OFFSET_PORT]
    payload_len = struct.unpack_from("<H", frame_bytes, r + OFFSET_PAYLOAD_LEN)[0]
    seq_num = frame_bytes[r + OFFSET_SEQ_NUM]

    fields = {
        "frame_len": orig_len,
        "direction": direction,
        "comm_id": comm_id,
        "port": port,
        "payload_len": payload_len,
        "seq_num": seq_num,
    }
    if with_signal:
        sig = radiotap_signal_dbm(frame_bytes)
        fields["signal_dbm"] = "" if sig is None else sig
    return fields


def convert_pcap(pcap_path: Path, output_path: Path, offset_ns: int = None,
                 with_signal: bool = False):
    """Convert a single PCAP to CSV; with offset_ns (epoch - monotonic of
    the capture host) also write host_timestamp_ns."""
    fieldnames = [
        "timestamp_s", "frame_len", "direction",
        "comm_id", "port", "payload_len", "seq_num"
    ]
    if offset_ns is not None:
        fieldnames.insert(0, "host_timestamp_ns")
    if with_signal:
        fieldnames.append("signal_dbm")

    n_frames = 0
    n_skipped = 0
    last_tx = (None, -1.0)  # (802.11 bytes, time) of the previous own frame

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for timestamp_s, frame_bytes, orig_len in parse_pcap_frames(pcap_path):
            fields = extract_db_fields(frame_bytes, orig_len, with_signal)
            if fields is None:
                n_skipped += 1
                continue
            if with_signal and fields["signal_dbm"] == "":
                body = frame_bytes[struct.unpack_from("<H", frame_bytes, 2)[0]:]
                if body == last_tx[0] and timestamp_s - last_tx[1] < TX_ECHO_WINDOW_S:
                    n_skipped += 1
                    last_tx = (None, -1.0)
                    continue
                last_tx = (body, timestamp_s)

            fields["timestamp_s"] = f"{timestamp_s:.9f}"
            if offset_ns is not None:
                fields["host_timestamp_ns"] = round(timestamp_s * 1e9) - offset_ns
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
    parser.add_argument("--clock-offset", type=Path, default=None,
                        help="clock_offset JSON of the capture host -> host_timestamp_ns")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="batch: only missions of this manifest (completed)")
    parser.add_argument("--host", default=None,
                        help="batch with --manifest: only rows with this host column")
    parser.add_argument("--onboard", action="store_true",
                        help="batch: pcap_onboard/ (wlan0) -> network_onboard/, with signal_dbm")
    parser.add_argument("--with-signal", action="store_true",
                        help="add the signal_dbm column (implied by --onboard)")
    args = parser.parse_args()
    offset_ns = None
    if args.clock_offset:
        offset_ns = int(json.loads(args.clock_offset.read_text())["epoch_minus_monotonic_ns"])

    if args.pcap:
        # Single file mode
        output = args.output or args.pcap.with_suffix(".csv")
        print(f"Converting {args.pcap.name}...")
        n_frames, n_skipped = convert_pcap(args.pcap, output, offset_ns, args.with_signal)
        print(f"  {n_frames} frames extracted, {n_skipped} skipped")
        print(f"  Output: {output}")

    elif args.dataset:
        # Batch mode
        pcap_dir = args.dataset / ("pcap_onboard" if args.onboard else "pcap")
        network_dir = args.dataset / ("network_onboard" if args.onboard else "network")
        network_dir.mkdir(exist_ok=True)

        pcaps = sorted(pcap_dir.glob("*.pcap"))
        if args.manifest:
            with open(args.manifest, newline="") as f:
                ids = {r["mission_id"] for r in csv.DictReader(f)
                       if r["status"] == "completed"
                       and (args.host is None or r.get("host") == args.host)}
            pcaps = [p for p in pcaps if p.stem in ids]
        if not pcaps:
            print(f"No pcap files in {pcap_dir}")
            sys.exit(1)

        print(f"Batch converting {len(pcaps)} PCAPs...")
        total_frames = 0
        total_skipped = 0

        for i, pcap_path in enumerate(pcaps):
            mission_id = pcap_path.stem
            output_path = network_dir / f"{mission_id}_network.csv"

            n_frames, n_skipped = convert_pcap(pcap_path, output_path, offset_ns,
                                               args.onboard or args.with_signal)
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