"""
Preprocessing for UAV telemetry missions.

Loads the five per-mission telemetry CSVs, aligns them onto a common
time grid (resampling), and provides normalization-statistics
computation/application.

Design decisions:
- Alignment key is `host_timestamp_ns` (clock on the capture host),
  shared across all message types of a mission.
- Two time axes (time_axis):
  - "raster" (default of the configs since 2026-11-03): fixed grid on the
    receiving host's clock, one row every 1/resample_freq_hz from the
    first common timestamp; each channel holds its last received value.
    This is what an IDS on the vehicle does: it infers on its own clock,
    whether or not new data arrived. A link loss therefore shows up as
    frozen values, and network frames received in that time still reach
    the model. Cost: arrival jitter or SITL slower than real time repeats
    or skips single samples.
  - "sample" (behaviour up to 2026-11-03, fallback when a config has no
    time_axis, e.g. older checkpoints): one reference message type
    defines the timesteps, all other types are matched to the nearest
    sample. A gap in the telemetry is a single step and invisible.
- The SITL recordings arrive at 4 Hz (MAVProxy's default stream rate
  overrides the logger's 50 Hz request a few seconds into each mission).
  In "sample" mode the initial 50 Hz segment is thinned out on the
  autopilot clock; the raster takes one value per tick anyway.
- Output column order is defined by features.get_feature_columns() and
  is identical for every mission (stable model input layout).
- Rows before the first 3D GPS fix are dropped. Some recordings start
  before the fix; GPS_RAW_INT then carries placeholder values (lat/lon 0,
  eph/epv 65535) that sit ~750 sigma off after normalization and caused
  the RSSM KL spikes of tuning study rssm_v2 (Befund in training_runs_4hz
  9.1). Only leading rows: a fix lost later in flight (GNSS jamming) is
  data and stays.
- Position representation (opt-in, default "abs"): "diff" replaces lat/lon
  of GPS_RAW_INT and GLOBAL_POSITION_INT by the change per timestep, north
  and east in metres; "rel_ekf" replaces the GPS lat/lon by the offset to
  the EKF position (GLOBAL_POSITION_INT), north and east in metres;
  "rel_start" replaces the lat/lon of both by north/east in metres relative
  to the first valid EKF position of the mission (same origin for both, so
  GPS - EKF is kept; leading rows without an EKF position are dropped). In
  absolute 1e-7 degrees a GNSS spoofing offset of ~30 m is ~0.1 sigma of
  the flight area and invisible (Vault bewertung_ids_explorativ, decision
  2026-11-09: "diff" for training, "rel_ekf" as ablation). Column names
  stay the same; the cache key contains the representation.
- Heading representation (opt-in, default "linear"): "sincos" replaces hdg
  (GLOBAL_POSITION_INT) and cog (GPS_RAW_INT), centidegrees, by their sine
  and cosine (features.HEADING_FEATURES). Linear, a turn through north is a
  jump over the whole range. The column names change (<name>_sin,
  <name>_cos); the cache key contains the representation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    HEADING_FEATURES,
    HEADING_REPRS,
    MESSAGE_FEATURES,
    NETWORK_FILE,
    NETWORK_TYPES,
    RADIO_TYPES,
    UPLINK_TYPES,
    UPLINK_DIRECTION,
    TIMESTAMP_COL,
    TIMESTAMP_FIELDS,
    get_feature_columns,
)

logger = logging.getLogger(__name__)

# Bumped whenever load_mission changes its output; part of the cache key
# (dataset._cache_dir), so stale caches are not reused.
# 2: rows before the first 3D GPS fix dropped.
# 3: on-board network CSVs regenerated without capture echoes (pcap_to_csv
#    echo filter over the whole window); the input data changed, not the code.
PREPROCESSING_VERSION = 4
TIME_AXES = ("raster", "sample")
GPS_FIX_COL = "GPS_RAW_INT__fix_type"
GPS_FIX_3D = 3
COUNT_STD_FLOOR = 1.0
POSITION_REPRS = ("abs", "diff", "rel_ekf", "rel_start")
POSITION_MSGS = ("GPS_RAW_INT", "GLOBAL_POSITION_INT")
M_PER_DEG = 111_320.0  # metres per degree of latitude (spherical earth)
RADIO_PREFIXES = tuple(f"{t}__" for t in sorted(RADIO_TYPES))


# ---------------------------------------------------------------------------
# Mission discovery
# ---------------------------------------------------------------------------

def discover_missions(
    telemetry_dir: str | Path,
    network_dir: str | Path | None = None,
    enabled_messages: dict[str, bool] | None = None,
) -> list[str]:
    """Return sorted mission IDs that have all required telemetry CSVs.

    A mission is complete iff every telemetry message type in
    MESSAGE_FEATURES has a file `{mission_id}_{MSG_TYPE}.csv` in
    telemetry_dir and, if a network stream is enabled, network_dir holds
    `{mission_id}_network.csv`.
    """
    telemetry_dir = Path(telemetry_dir)
    candidates: dict[str, set[str]] = {}

    for f in telemetry_dir.glob("mission_*_*.csv"):
        # filename: mission_00001_GLOBAL_POSITION_INT.csv
        stem = f.stem  # mission_00001_GLOBAL_POSITION_INT
        parts = stem.split("_", 2)  # ["mission", "00001", "GLOBAL_POSITION_INT"]
        if len(parts) != 3:
            continue
        mission_id = f"{parts[0]}_{parts[1]}"
        msg_type = parts[2]
        if msg_type in MESSAGE_FEATURES and msg_type not in NETWORK_TYPES:
            candidates.setdefault(mission_id, set()).add(msg_type)

    required = set(MESSAGE_FEATURES.keys()) - NETWORK_TYPES
    complete = sorted(m for m, types in candidates.items() if types >= required)
    enabled_net = [t for t in NETWORK_TYPES
                   if enabled_messages and enabled_messages.get(t, False)]
    if enabled_net:
        if network_dir is None:
            raise ValueError("network stream enabled but no network_dir given")
        complete = [m for m in complete
                    if (Path(network_dir) / f"{m}_{NETWORK_FILE}.csv").exists()]

    n_incomplete = len(candidates) - len(complete)
    if n_incomplete:
        logger.warning("%d missions skipped (missing message types)", n_incomplete)
    return complete


# ---------------------------------------------------------------------------
# Per-mission loading + alignment
# ---------------------------------------------------------------------------

# Minimum spacing of kept reference samples, as a fraction of the target
# period. Below 1.0 so that native samples at exactly the target rate
# (and slight autopilot jitter) are never dropped; the 50 Hz start
# segment is thinned to every 12th sample (240 ms at 4 Hz).
_MIN_SPACING_FRACTION = 0.9


def _autopilot_time_ns(df: pd.DataFrame, msg_type: str) -> np.ndarray:
    """Autopilot clock of a message type in ns, host clock as fallback.

    Recordings without an autopilot timestamp (e.g. converted FlyPaw
    data) only carry the host timestamp.
    """
    fields = TIMESTAMP_FIELDS[msg_type]
    if not fields or fields[0] not in df.columns:
        return df[TIMESTAMP_COL].to_numpy(dtype=np.int64)
    field = fields[0]
    scale = 1_000_000 if field.endswith("_ms") else 1_000
    return df[field].to_numpy(dtype=np.int64) * scale


def _thin_out(t_ns: np.ndarray, min_spacing_ns: int) -> np.ndarray:
    """Boolean mask keeping samples at least min_spacing_ns apart.

    A backwards jump of the clock (autopilot reboot) restarts the
    spacing instead of suppressing all following samples.
    """
    keep = np.zeros(len(t_ns), dtype=bool)
    last = None
    for i, t in enumerate(t_ns):
        if last is None or t - last >= min_spacing_ns or t < last:
            keep[i] = True
            last = t
    return keep


def load_mission(
    telemetry_dir: str | Path,
    mission_id: str,
    resample_freq_hz: float = 4.0,
    enabled_messages: dict[str, bool] | None = None,
    network_dir: str | Path | None = None,
    time_axis: str = "sample",
    position_repr: str = "abs",
    heading_repr: str = "linear",
) -> pd.DataFrame:
    """Load one mission and align all message types onto a common time axis.

    time_axis "raster": rows on a fixed grid of the host clock, every
    message type contributes its last sample at or before the tick.
    time_axis "sample": the first enabled message type is the reference:
    its samples, thinned to ~resample_freq_hz on the autopilot clock, are
    the timesteps. Every other message type contributes its nearest sample
    within half a period, so no sample is repeated as long as all types
    arrive at the target rate.

    Args:
        telemetry_dir: Directory containing the per-mission CSVs.
        mission_id: e.g. "mission_00001".
        resample_freq_hz: Target cadence. Should not exceed the native
            rate of the recordings, otherwise nothing is thinned and the
            native rate is kept.
        enabled_messages: Message-type toggle dict (None = all enabled).
        network_dir: Directory with `{mission_id}_network.csv`
            (pcap frames, host_timestamp_ns on the telemetry clock), source
            of the network_up / network_down streams.
        time_axis: "raster" or "sample", see module doc.
        position_repr: "abs", "diff" or "rel_ekf", see module doc.
        heading_repr: "linear" or "sincos", see module doc.

    Returns:
        DataFrame indexed 0..T-1 with columns
        [t_ns] + get_feature_columns(enabled_messages, heading_repr).
        Feature columns are float32.

    Raises:
        FileNotFoundError: if a required CSV is missing.
        ValueError: if the overlapping time range is empty.
    """
    telemetry_dir = Path(telemetry_dir)
    if time_axis not in TIME_AXES:
        raise ValueError(f"time_axis must be one of {TIME_AXES}, got {time_axis!r}")
    if position_repr not in POSITION_REPRS:
        raise ValueError(f"position_repr must be one of {POSITION_REPRS}, got {position_repr!r}")
    if heading_repr not in HEADING_REPRS:
        raise ValueError(f"heading_repr must be one of {HEADING_REPRS}, got {heading_repr!r}")
    if enabled_messages is None:
        enabled_messages = {k: k not in RADIO_TYPES for k in MESSAGE_FEATURES}

    period_ns = int(1e9 / resample_freq_hz)

    frames: dict[str, pd.DataFrame] = {}
    for msg_type, feats in MESSAGE_FEATURES.items():
        if not enabled_messages.get(msg_type, False):
            continue
        if msg_type in NETWORK_TYPES:
            path = Path(network_dir) / f"{mission_id}_{NETWORK_FILE}.csv"
        else:
            path = telemetry_dir / f"{mission_id}_{msg_type}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        if msg_type in RADIO_TYPES:
            if time_axis != "raster":
                raise ValueError(f"{msg_type} needs time_axis raster")
            wanted = {TIMESTAMP_COL, "direction", "port", "payload_len", "seq_num"}
        else:
            wanted = {TIMESTAMP_COL, *TIMESTAMP_FIELDS[msg_type], *feats}
        df = pd.read_csv(path, usecols=lambda c: c in wanted)
        if msg_type in NETWORK_TYPES:
            up = df["direction"] == UPLINK_DIRECTION
            df = df[up if msg_type in UPLINK_TYPES else ~up]
            if df.empty:
                raise ValueError(f"{mission_id}: no frames for {msg_type}")
        if msg_type in RADIO_TYPES:
            frames[msg_type] = _radio_frames(df, msg_type)
            continue
        df = df.sort_values(TIMESTAMP_COL).drop_duplicates(
            subset=TIMESTAMP_COL, keep="last"
        )
        df = df.rename(columns={f: f"{msg_type}__{f}" for f in feats})
        frames[msg_type] = df

    if not frames:
        raise ValueError("No message types enabled")

    # Overlapping time range across all enabled message types.
    t_start = max(int(df[TIMESTAMP_COL].iloc[0]) for df in frames.values())
    t_end = min(int(df[TIMESTAMP_COL].iloc[-1]) for df in frames.values())
    if t_end <= t_start:
        raise ValueError(f"{mission_id}: empty overlapping time range")

    if time_axis == "raster":
        aligned = _align_raster(frames, t_start, t_end, period_ns)
    else:
        aligned = _align_samples(frames, t_start, t_end, period_ns)

    feature_cols = get_feature_columns(enabled_messages)
    aligned[feature_cols] = aligned[feature_cols].ffill().bfill()

    if aligned[feature_cols].isna().any().any():
        raise ValueError(f"{mission_id}: NaNs remain after fill")

    if GPS_FIX_COL in aligned.columns:
        fixed = np.flatnonzero(aligned[GPS_FIX_COL].to_numpy() >= GPS_FIX_3D)
        if fixed.size == 0:
            raise ValueError(f"{mission_id}: no 3D GPS fix")
        aligned = aligned.iloc[fixed[0]:].reset_index(drop=True)

    if position_repr == "rel_start":
        e = POSITION_MSGS[1]
        if f"{e}__lat" not in aligned.columns:
            raise ValueError("position_repr rel_start needs GLOBAL_POSITION_INT")
        valid = np.flatnonzero((aligned[f"{e}__lat"].to_numpy() != 0) & (aligned[f"{e}__lon"].to_numpy() != 0))
        if valid.size == 0:
            raise ValueError(f"{mission_id}: no EKF position")
        aligned = aligned.iloc[valid[0]:].reset_index(drop=True)

    # Before the float32 cast: lat/lon in 1e-7 degrees only resolve ~0.4 m there.
    _apply_position_repr(aligned, position_repr)
    _apply_heading_repr(aligned, heading_repr, mission_id)

    out_cols = get_feature_columns(enabled_messages, heading_repr)
    aligned = aligned.rename(columns={TIMESTAMP_COL: "t_ns"})
    aligned[out_cols] = aligned[out_cols].astype(np.float32)
    return aligned[["t_ns"] + out_cols]


def _apply_heading_repr(aligned: pd.DataFrame, heading_repr: str, mission_id: str) -> None:
    """heading_repr "sincos": add <col>_sin / <col>_cos for the heading channels (the linear column is not selected
    any more). Values above 360.00 deg (MAVLink UINT16_MAX = unknown) are not in the recorded data and raise."""
    if heading_repr != "sincos":
        return
    for msg, feat in HEADING_FEATURES.items():
        col = f"{msg}__{feat}"
        if col not in aligned.columns:
            continue
        cdeg = aligned[col].to_numpy(np.float64)
        if (cdeg > 36000).any() or (cdeg < 0).any():
            raise ValueError(f"{mission_id}: {col} outside 0..36000 centidegrees")
        rad = np.deg2rad(cdeg / 100.0)
        aligned[f"{col}_sin"] = np.sin(rad)
        aligned[f"{col}_cos"] = np.cos(rad)


def _north_east_m(dlat: np.ndarray, dlon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differences of lat/lon (1e-7 degrees) as north/east in metres at latitude lat."""
    k = 1e-7 * M_PER_DEG
    return dlat * k, dlon * k * np.cos(np.deg2rad(lat * 1e-7))


def _apply_position_repr(aligned: pd.DataFrame, position_repr: str) -> None:
    """Rewrite the lat/lon columns in place, see module doc."""
    if position_repr == "diff":
        for msg in POSITION_MSGS:
            la, lo = f"{msg}__lat", f"{msg}__lon"
            if la not in aligned.columns:
                continue
            lat = aligned[la].to_numpy(np.float64)
            lon = aligned[lo].to_numpy(np.float64)
            aligned[la], aligned[lo] = _north_east_m(
                np.diff(lat, prepend=lat[:1]), np.diff(lon, prepend=lon[:1]), lat)
    elif position_repr == "rel_ekf":
        g, e = POSITION_MSGS
        cols = [f"{g}__lat", f"{g}__lon", f"{e}__lat", f"{e}__lon"]
        if not all(c in aligned.columns for c in cols):
            raise ValueError("position_repr rel_ekf needs GPS_RAW_INT and GLOBAL_POSITION_INT")
        glat, glon, elat, elon = (aligned[c].to_numpy(np.float64) for c in cols)
        aligned[cols[0]], aligned[cols[1]] = _north_east_m(glat - elat, glon - elon, elat)
    elif position_repr == "rel_start":
        e = POSITION_MSGS[1]
        lat0 = float(aligned[f"{e}__lat"].iloc[0]); lon0 = float(aligned[f"{e}__lon"].iloc[0])
        for msg in POSITION_MSGS:
            la, lo = f"{msg}__lat", f"{msg}__lon"
            if la not in aligned.columns:
                continue
            lat = aligned[la].to_numpy(np.float64); lon = aligned[lo].to_numpy(np.float64)
            aligned[la], aligned[lo] = _north_east_m(lat - lat0, lon - lon0, np.full_like(lat, lat0))


def _feature_cols_of(df: pd.DataFrame, msg_type: str) -> list[str]:
    return [TIMESTAMP_COL] + [c for c in df.columns if c.startswith(f"{msg_type}__")]


def _align_raster(frames: dict[str, pd.DataFrame], t_start: int, t_end: int,
                  period_ns: int) -> pd.DataFrame:
    """Fixed host-clock grid, last received value of every message type.

    t_start is the latest first timestamp over all types, so every type has
    a sample at or before the first tick.
    """
    aligned = pd.DataFrame({TIMESTAMP_COL: np.arange(t_start, t_end + 1, period_ns,
                                                     dtype=np.int64)})
    for msg_type, df in frames.items():
        if msg_type in RADIO_TYPES:
            aligned = _bin_radio(aligned, df, msg_type, period_ns)
            continue
        aligned = pd.merge_asof(aligned, df[_feature_cols_of(df, msg_type)],
                                on=TIMESTAMP_COL, direction="backward")
    return aligned


def _radio_frames(df: pd.DataFrame, msg_type: str) -> pd.DataFrame:
    """Per-frame values of a radio_* stream: payload bytes and the frames
    lost before it (gap of seq_num mod 256 to the previous frame of the same
    port, 0 for the first frame of a port)."""
    df = df.sort_values(TIMESTAMP_COL, kind="stable")
    step = df.groupby("port")["seq_num"].diff().fillna(1).to_numpy() % 256
    lost = np.where(step == 0, 0, step - 1)
    return pd.DataFrame({
        TIMESTAMP_COL: df[TIMESTAMP_COL].to_numpy(np.int64),
        f"{msg_type}__frames": 1.0,
        f"{msg_type}__bytes": df["payload_len"].to_numpy(np.float64),
        f"{msg_type}__payload_len_max": df["payload_len"].to_numpy(np.float64),
        f"{msg_type}__seq_lost": lost.astype(np.float64),
    })


def _bin_radio(aligned: pd.DataFrame, df: pd.DataFrame, msg_type: str,
               period_ns: int) -> pd.DataFrame:
    """Sum (max for payload_len_max) of the frames in (t - period, t] for
    every tick t; ticks without frames are 0."""
    ticks = aligned[TIMESTAMP_COL].to_numpy(np.int64)
    t = df[TIMESTAMP_COL].to_numpy(np.int64)
    k = np.searchsorted(ticks, t, side="left")  # ticks[k-1] < t <= ticks[k]
    ok = (k < len(ticks)) & (t > ticks[0] - period_ns)
    k = k[ok]
    out = aligned.copy()
    for f in ("frames", "bytes", "seq_lost"):
        col = f"{msg_type}__{f}"
        out[col] = np.bincount(k, weights=df[col].to_numpy()[ok], minlength=len(ticks))
    col = f"{msg_type}__payload_len_max"
    mx = np.zeros(len(ticks))
    np.maximum.at(mx, k, df[col].to_numpy()[ok])
    out[col] = mx
    return out


def _align_samples(frames: dict[str, pd.DataFrame], t_start: int, t_end: int,
                   period_ns: int) -> pd.DataFrame:
    """Sample-driven axis: reference type thinned to the target rate, other
    types matched to the nearest sample within half a period."""
    ref_type = next(iter(frames))
    ref = frames[ref_type]
    ref = ref[(ref[TIMESTAMP_COL] >= t_start) & (ref[TIMESTAMP_COL] <= t_end)]
    keep = _thin_out(
        _autopilot_time_ns(ref, ref_type),
        int(_MIN_SPACING_FRACTION * period_ns),
    )
    ref_cols = [TIMESTAMP_COL] + [
        c for c in ref.columns if c.startswith(f"{ref_type}__")
    ]
    aligned = ref.loc[keep, ref_cols].reset_index(drop=True)

    # Nearest sample within half a period; misses stay NaN and are
    # forward-filled afterwards (dropouts are rare in SITL data).
    for msg_type, df in frames.items():
        if msg_type == ref_type:
            continue
        cols = [TIMESTAMP_COL] + [
            c for c in df.columns if c.startswith(f"{msg_type}__")
        ]
        aligned = pd.merge_asof(
            aligned,
            df[cols],
            on=TIMESTAMP_COL,
            direction="nearest",
            tolerance=period_ns // 2,
        )
    return aligned


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """Per-feature z-score normalization with persistable statistics.

    Statistics must be computed on the training split only and are then
    applied unchanged to val/test (and later to attack data).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, columns: list[str],
                 train_ids_sha1: str | None = None):
        # Fingerprint of the training missions the statistics come from (None in files from
        # before 2026-11-26): a cache that grows (v4 while it is being flown) refits when it changes.
        self.train_ids_sha1 = train_ids_sha1
        self.mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        # Constant features (std == 0) are mapped to 0 instead of NaN.
        std = np.where(std < 1e-12, 1.0, std)
        # Count streams (radio_*: frames, bytes, payload_len_max, seq_lost):
        # standard deviation at least one unit (1 frame / 1 byte), so a single
        # lost or extra frame is at most 1 sigma even where it is rare in the
        # normal data (decision 2026-11-08; log1p was rejected).
        floor = np.array([COUNT_STD_FLOOR if c.startswith(RADIO_PREFIXES) else 0.0
                          for c in columns], dtype=np.float32)
        self.std = np.maximum(std, floor)
        self.columns = columns

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalize array of shape (..., n_features)."""
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    # -- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        if self.train_ids_sha1:
            payload["train_ids_sha1"] = self.train_ids_sha1
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        payload = json.loads(Path(path).read_text())
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
            columns=payload["columns"],
            train_ids_sha1=payload.get("train_ids_sha1"),
        )

    # -- fitting ---------------------------------------------------------

    @classmethod
    def fit(cls, arrays: list[np.ndarray], columns: list[str]) -> "Normalizer":
        """Fit on a list of (T_i, n_features) arrays via streaming moments.

        Avoids concatenating all missions in memory.
        """
        n_features = arrays[0].shape[1]
        count = 0
        s1 = np.zeros(n_features, dtype=np.float64)
        s2 = np.zeros(n_features, dtype=np.float64)
        for a in arrays:
            a64 = a.astype(np.float64)
            count += a.shape[0]
            s1 += a64.sum(axis=0)
            s2 += (a64 ** 2).sum(axis=0)
        mean = s1 / count
        var = s2 / count - mean ** 2
        std = np.sqrt(np.clip(var, 0.0, None))
        return cls(mean=mean, std=std, columns=columns)