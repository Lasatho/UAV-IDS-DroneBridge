"""
Preprocessing for UAV telemetry missions.

Loads the five per-mission telemetry CSVs, aligns them onto a common
time grid (resampling), and provides normalization-statistics
computation/application.

Design decisions:
- Alignment key is `host_timestamp_ns` (clock on the capture host),
  shared across all message types of a mission.
- Each message type arrives at its own rate. All types are resampled
  onto a uniform grid via merge_asof (last-observation-carried-forward,
  bounded by a tolerance) so that one row = one timestep with the full
  feature vector.
- Output column order is defined by features.get_feature_columns() and
  is identical for every mission (stable model input layout).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    MESSAGE_FEATURES,
    TIMESTAMP_COL,
    get_feature_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mission discovery
# ---------------------------------------------------------------------------

def discover_missions(telemetry_dir: str | Path) -> list[str]:
    """Return sorted mission IDs that have all required telemetry CSVs.

    A mission is complete iff every message type in MESSAGE_FEATURES has
    a file `{mission_id}_{MSG_TYPE}.csv` in telemetry_dir.
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
        if msg_type in MESSAGE_FEATURES:
            candidates.setdefault(mission_id, set()).add(msg_type)

    required = set(MESSAGE_FEATURES.keys())
    complete = sorted(m for m, types in candidates.items() if types >= required)

    n_incomplete = len(candidates) - len(complete)
    if n_incomplete:
        logger.warning("%d missions skipped (missing message types)", n_incomplete)
    return complete


# ---------------------------------------------------------------------------
# Per-mission loading + alignment
# ---------------------------------------------------------------------------

def load_mission(
    telemetry_dir: str | Path,
    mission_id: str,
    resample_freq_hz: float = 10.0,
    enabled_messages: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Load one mission and align all message types onto a uniform grid.

    Args:
        telemetry_dir: Directory containing the per-mission CSVs.
        mission_id: e.g. "mission_00001".
        resample_freq_hz: Target grid frequency.
        enabled_messages: Message-type toggle dict (None = all enabled).

    Returns:
        DataFrame indexed 0..T-1 with columns
        [t_ns] + get_feature_columns(enabled_messages).
        Feature columns are float32.

    Raises:
        FileNotFoundError: if a required CSV is missing.
        ValueError: if the overlapping time range is empty.
    """
    telemetry_dir = Path(telemetry_dir)
    if enabled_messages is None:
        enabled_messages = {k: True for k in MESSAGE_FEATURES}

    period_ns = int(1e9 / resample_freq_hz)

    frames: dict[str, pd.DataFrame] = {}
    for msg_type, feats in MESSAGE_FEATURES.items():
        if not enabled_messages.get(msg_type, False):
            continue
        path = telemetry_dir / f"{mission_id}_{msg_type}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path, usecols=[TIMESTAMP_COL] + feats)
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

    grid = pd.DataFrame(
        {TIMESTAMP_COL: np.arange(t_start, t_end + 1, period_ns, dtype=np.int64)}
    )

    # merge_asof: for each grid point take the most recent observation.
    # Tolerance = 2 grid periods; beyond that the value is NaN and we
    # forward-fill afterwards (sensor dropouts are not expected in SITL data).
    aligned = grid
    for msg_type, df in frames.items():
        aligned = pd.merge_asof(
            aligned,
            df,
            on=TIMESTAMP_COL,
            direction="backward",
            tolerance=2 * period_ns,
        )

    feature_cols = get_feature_columns(enabled_messages)
    aligned[feature_cols] = aligned[feature_cols].ffill().bfill()

    if aligned[feature_cols].isna().any().any():
        raise ValueError(f"{mission_id}: NaNs remain after fill")

    aligned = aligned.rename(columns={TIMESTAMP_COL: "t_ns"})
    aligned[feature_cols] = aligned[feature_cols].astype(np.float32)
    return aligned[["t_ns"] + feature_cols]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """Per-feature z-score normalization with persistable statistics.

    Statistics must be computed on the training split only and are then
    applied unchanged to val/test (and later to attack data).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, columns: list[str]):
        self.mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        # Constant features (std == 0) are mapped to 0 instead of NaN.
        self.std = np.where(std < 1e-12, 1.0, std)
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
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        payload = json.loads(Path(path).read_text())
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
            columns=payload["columns"],
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