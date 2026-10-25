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
}

# Streams read from {mission_id}_network.csv in the network directory,
# selected by the direction byte: 0x01 = to the drone (uplink); everything
# else (0x03 = to the GCS, and frames that are no valid DroneBridge frame)
# goes to the downlink stream so foreign frames stay visible.
NETWORK_FILE = "network"
NETWORK_TYPES = {"network_up", "network_down"}
UPLINK_DIRECTION = 1

# Timestamp columns to drop per message type (not used as features)
TIMESTAMP_FIELDS: dict[str, list[str]] = {
    "GLOBAL_POSITION_INT": ["time_boot_ms"],
    "GPS_RAW_INT": ["time_usec"],
    "RAW_IMU": ["time_usec"],
    "SCALED_PRESSURE": ["time_boot_ms"],
    "SERVO_OUTPUT_RAW": ["time_usec"],
    "network_up": [],  # host clock only
    "network_down": [],
}


def get_feature_columns(enabled_messages: dict[str, bool] | None = None) -> list[str]:
    """Return ordered list of feature column names for enabled message types.

    Column names are prefixed with message type to avoid collisions
    (e.g., GPS_RAW_INT has 'lat' and GLOBAL_POSITION_INT has 'lat').

    Args:
        enabled_messages: Dict mapping message type to bool.
            If None, all message types are enabled.

    Returns:
        List of prefixed feature column names in stable order.
    """
    if enabled_messages is None:
        enabled_messages = {k: True for k in MESSAGE_FEATURES}

    columns = []
    for msg_type, features in MESSAGE_FEATURES.items():
        if enabled_messages.get(msg_type, False):
            columns.extend(f"{msg_type}__{feat}" for feat in features)
    return columns


def get_num_features(enabled_messages: dict[str, bool] | None = None) -> int:
    """Return total number of features for enabled message types."""
    return len(get_feature_columns(enabled_messages))