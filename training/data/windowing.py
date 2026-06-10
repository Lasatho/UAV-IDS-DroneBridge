"""
Sliding-window extraction over aligned mission time series.

A window index is a (mission_idx, start) pair; the actual tensor slice
is materialized lazily in the Dataset to keep memory flat.
"""

from __future__ import annotations

import numpy as np


def compute_window_starts(
    series_length: int,
    window_length: int,
    stride: int,
) -> np.ndarray:
    """Start indices of all full windows in a series of given length.

    Windows are fully contained (no padding); a mission shorter than
    window_length yields zero windows.
    """
    if series_length < window_length:
        return np.empty(0, dtype=np.int64)
    n = 1 + (series_length - window_length) // stride
    return np.arange(n, dtype=np.int64) * stride


def build_window_index(
    series_lengths: list[int],
    window_length: int,
    stride: int,
) -> np.ndarray:
    """Global window index over multiple missions.

    Args:
        series_lengths: Length T_i of each mission's aligned series.
        window_length: Timesteps per window.
        stride: Stride between consecutive window starts.

    Returns:
        Array of shape (N_windows, 2): [mission_idx, start].
    """
    entries = []
    for m_idx, length in enumerate(series_lengths):
        starts = compute_window_starts(length, window_length, stride)
        if starts.size:
            entries.append(
                np.stack(
                    [np.full_like(starts, m_idx), starts],
                    axis=1,
                )
            )
    if not entries:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(entries, axis=0)