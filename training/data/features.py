"""
Feature definitions for UAV telemetry message types.

Central source of truth for column names, timestamp fields,
and feature ordering. All downstream code references this module.
"""

# Timestamp column present in all CSVs (alignment key)
TIMESTAMP_COL = "host_timestamp_ns"

# Per-message-type: columns that are sensor features (excludes timestamps)
MESSAGE_FEATURES: dict[str, list[str]] = {
    "GLOBAL_POSITION_INT": [
        "lat", "lon", "alt", "relative_alt", "vx", "vy", "vz", "hdg",
    ],
    "GPS_RAW_INT": [
        "fix_type", "lat", "lon", "alt", "eph", "epv", "vel", "cog",
        "satellites_visible",
    ],
    "RAW_IMU": [
        "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro",
        "xmag", "ymag", "zmag",
    ],
    "SCALED_PRESSURE": [
        "press_abs", "press_diff", "temperature",
    ],
    "SERVO_OUTPUT_RAW": [
        "servo1_raw", "servo2_raw", "servo3_raw", "servo4_raw",
        "servo5_raw", "servo6_raw", "servo7_raw", "servo8_raw",
    ],
    # DroneBridge radio frames from the pcap capture, one row per frame
    # with the raw header fields (data/pcap_to_csv.py), split by direction
    # into two streams that are each treated exactly like a telemetry
    # message type. One mixed stream would mostly lose the uplink on the
    # 4 Hz grid (~1 uplink vs ~5.5 downlink frames/s).
    "network_up": [
        "frame_len", "direction", "comm_id", "port", "payload_len", "seq_num",
    ],
    "network_down": [
        "frame_len", "direction", "comm_id", "port", "payload_len", "seq_num",
    ],
    # The same frames summarized per raster tick instead of the header
    # fields of the last frame (time_axis "raster" only): frames and payload
    # bytes received in (t - period, t], largest payload, and frames lost
    # according to the DroneBridge sequence number (gaps of seq_num, mod
    # 256, per port). An empty tick is 0, so missing traffic (rf_jamming) is
    # visible; the last-value fill of network_* hides it.
    "radio_up": [
        "frames", "bytes", "payload_len_max", "seq_lost",
    ],
    "radio_down": [
        "frames", "bytes", "payload_len_max", "seq_lost",
    ],
}

# Streams read from {mission_id}_network.csv in the network directory,
# selected by the direction byte: 0x01 = to the drone (uplink); everything
# else (0x03 = to the GCS, and frames that are no valid DroneBridge frame)
# goes to the downlink stream so foreign frames stay visible.
NETWORK_FILE = "network"
RADIO_TYPES = {"radio_up", "radio_down"}  # opt-in, not part of the default (None) set
NETWORK_TYPES = {"network_up", "network_down"} | RADIO_TYPES
UPLINK_TYPES = {"network_up", "radio_up"}
UPLINK_DIRECTION = 1

# Circular channels in centidegrees (0..35999). heading_repr "sincos" (opt-in, default "linear") replaces each by its
# sine and cosine, columns <name>_sin and <name>_cos at the same place. Linear, a turn through north jumps by the
# whole range (359.5 deg -> 0.2 deg), which the models see as a huge change (10.5 % of the normal windows, Vault
# modellumbau_rssm_mtsjepa 2026-11-20).
HEADING_FEATURES = {"GLOBAL_POSITION_INT": "hdg", "GPS_RAW_INT": "cog"}
HEADING_REPRS = ("linear", "sincos")

# Timestamp columns to drop per message type (not used as features)
TIMESTAMP_FIELDS: dict[str, list[str]] = {
    "GLOBAL_POSITION_INT": ["time_boot_ms"],
    "GPS_RAW_INT": ["time_usec"],
    "RAW_IMU": ["time_usec"],
    "SCALED_PRESSURE": ["time_boot_ms"],
    "SERVO_OUTPUT_RAW": ["time_usec"],
    "network_up": [],  # host clock only
    "network_down": [],
    "radio_up": [],
    "radio_down": [],
}


def get_feature_columns(enabled_messages: dict[str, bool] | None = None,
                        heading_repr: str = "linear") -> list[str]:
    """Return ordered list of feature column names for enabled message types.

    Column names are prefixed with message type to avoid collisions
    (e.g., GPS_RAW_INT has 'lat' and GLOBAL_POSITION_INT has 'lat').

    Args:
        enabled_messages: Dict mapping message type to bool.
            If None, all message types are enabled.
        heading_repr: "linear" or "sincos" (HEADING_FEATURES as two columns).

    Returns:
        List of prefixed feature column names in stable order.
    """
    if enabled_messages is None:
        enabled_messages = {k: k not in RADIO_TYPES for k in MESSAGE_FEATURES}

    if heading_repr not in HEADING_REPRS:
        raise ValueError(f"heading_repr must be one of {HEADING_REPRS}, got {heading_repr!r}")
    columns = []
    for msg_type, features in MESSAGE_FEATURES.items():
        if enabled_messages.get(msg_type, False):
            for feat in features:
                if heading_repr == "sincos" and HEADING_FEATURES.get(msg_type) == feat:
                    columns.extend((f"{msg_type}__{feat}_sin", f"{msg_type}__{feat}_cos"))
                else:
                    columns.append(f"{msg_type}__{feat}")
    return columns


def get_num_features(enabled_messages: dict[str, bool] | None = None,
                     heading_repr: str = "linear") -> int:
    """Return total number of features for enabled message types."""
    return len(get_feature_columns(enabled_messages, heading_repr))