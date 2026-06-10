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
}

# Timestamp columns to drop per message type (not used as features)
TIMESTAMP_FIELDS: dict[str, list[str]] = {
    "GLOBAL_POSITION_INT": ["time_boot_ms"],
    "GPS_RAW_INT": ["time_usec"],
    "RAW_IMU": ["time_usec"],
    "SCALED_PRESSURE": ["time_boot_ms"],
    "SERVO_OUTPUT_RAW": ["time_usec"],
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