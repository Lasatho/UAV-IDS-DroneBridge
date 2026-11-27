"""
Streaming (on-board) counterpart of data.preprocessing.load_mission for time_axis "raster" (WP7, Vault
Planung/mcp_werkzeuge E4, 2026-11-26). Consumes telemetry messages and radio frames in arrival order and emits one
feature row per tick of its own clock, using only data received up to that tick:

- ticks every 1/freq from the first instant at which every enabled stream has delivered a sample (offline: the
  latest first timestamp over the streams);
- each telemetry channel holds its last value at or before the tick (offline: merge_asof backward);
- radio_*: frames, bytes, largest payload, frames lost by the DroneBridge sequence number (gap mod 256 to the
  previous frame of the same port) over (t - period, t] (offline: _radio_frames + _bin_radio);
- no rows before the first tick with a 3D GPS fix and, for rel_start, before the first tick with an EKF position
  (offline: leading rows dropped);
- position_repr rel_start: north/east metres relative to the EKF position of the first emitted row; heading_repr
  sincos: sine and cosine of hdg/cog; output in the column order of features.get_feature_columns, float32.

replay_mission() feeds the recorded CSVs of one mission through it, for the parity check against load_mission.
Only the representations used on v3/v4 are implemented (raster, rel_start or abs, linear or sincos).
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (HEADING_FEATURES, MESSAGE_FEATURES, NETWORK_FILE, NETWORK_TYPES, RADIO_TYPES, TIMESTAMP_COL,
                       UPLINK_DIRECTION, UPLINK_TYPES, get_feature_columns)
from .preprocessing import GPS_FIX_3D, GPS_FIX_COL, M_PER_DEG, POSITION_MSGS


class OnboardPreprocessor:
    def __init__(self, enabled_messages: dict[str, bool], freq_hz: float = 4.0, position_repr: str = "rel_start",
                 heading_repr: str = "sincos"):
        if position_repr not in ("abs", "rel_start") or heading_repr not in ("linear", "sincos"):
            raise NotImplementedError("streaming: position_repr abs/rel_start, heading_repr linear/sincos")
        self.enabled = {k: v for k, v in enabled_messages.items() if v}
        self.tel = [t for t in self.enabled if t not in NETWORK_TYPES]
        self.radio = [t for t in self.enabled if t in RADIO_TYPES]
        if any(t in NETWORK_TYPES and t not in RADIO_TYPES for t in self.enabled):
            raise NotImplementedError("streaming: network_* (last frame header) not implemented, use radio_*")
        self.period = int(1e9 / freq_hz)
        self.position_repr, self.heading_repr = position_repr, heading_repr
        self.cols = get_feature_columns(enabled_messages, heading_repr)
        self.last: dict[str, dict] = {}          # telemetry type -> last values
        self.first_seen: dict[str, int] = {}     # stream -> first timestamp
        self.bins: dict[str, list] = {t: [] for t in self.radio}   # frames (ts, payload, lost) not yet binned
        self.last_seq: dict[tuple, int] = {}     # (stream, port) -> last seq_num
        self.next_tick: int | None = None
        self.started = False; self.fixed = False; self.origin = None
        self.max_tick: int | None = None  # replay: last tick (offline grid ends at the earliest last timestamp)

    # -- input ---------------------------------------------------------------
    def feed_message(self, msg_type: str, ts_ns: int, values: dict) -> list[np.ndarray]:
        out = self._advance(ts_ns)
        self.last[msg_type] = values; self.first_seen.setdefault(msg_type, ts_ns)
        return out

    def feed_frame(self, ts_ns: int, direction: int, port: int, payload_len: float, seq_num: int) -> list[np.ndarray]:
        out = self._advance(ts_ns)
        stream = "radio_up" if direction == UPLINK_DIRECTION else "radio_down"
        if stream in self.radio:
            key = (stream, port); prev = self.last_seq.get(key)
            step = 1 if prev is None else (seq_num - prev) % 256
            lost = 0 if step == 0 else step - 1
            self.last_seq[key] = seq_num
            self.bins[stream].append((ts_ns, float(payload_len), float(lost)))
            self.first_seen.setdefault(stream, ts_ns)
        return out

    # -- clock ---------------------------------------------------------------
    def _advance(self, ts_ns: int) -> list[np.ndarray]:
        """Emit every tick strictly before ts_ns (a sample at exactly a tick belongs to it: emitted on the next input)."""
        rows = []
        if self.next_tick is None:
            need = self.tel + self.radio
            if all(s in self.first_seen for s in need):
                self.next_tick = max(self.first_seen[s] for s in need)
            else:
                return rows
        while self.next_tick < ts_ns and (self.max_tick is None or self.next_tick <= self.max_tick):
            r = self._tick(self.next_tick)
            if r is not None:
                rows.append(r)
            self.next_tick += self.period
        return rows

    def flush(self, until_ns: int) -> list[np.ndarray]:
        return self._advance(until_ns + 1)

    def _tick(self, t: int):
        row = {}
        for m in self.tel:
            for f in MESSAGE_FEATURES[m]:
                row[f"{m}__{f}"] = float(self.last[m][f])
        first = not self.started
        for s in self.radio:
            lo = t - self.period
            keep, cnt, byt, mx, lost = [], 0.0, 0.0, 0.0, 0.0
            for ts, pl, ls in self.bins[s]:
                if ts > t:
                    keep.append((ts, pl, ls)); continue
                if ts > lo or (first and ts > lo):
                    cnt += 1; byt += pl; mx = max(mx, pl); lost += ls
            self.bins[s] = keep
            row.update({f"{s}__frames": cnt, f"{s}__bytes": byt, f"{s}__payload_len_max": mx, f"{s}__seq_lost": lost})
        self.started = True
        if not self.fixed:
            if GPS_FIX_COL in row and row[GPS_FIX_COL] < GPS_FIX_3D:
                return None
            self.fixed = True
        if self.position_repr == "rel_start":
            e = POSITION_MSGS[1]
            if self.origin is None:
                if row[f"{e}__lat"] == 0 or row[f"{e}__lon"] == 0:
                    return None
                self.origin = (row[f"{e}__lat"], row[f"{e}__lon"])
            lat0, lon0 = self.origin; k = 1e-7 * M_PER_DEG
            for msg in POSITION_MSGS:
                la, lo_ = f"{msg}__lat", f"{msg}__lon"
                if la in row:
                    row[la], row[lo_] = (row[la] - lat0) * k, (row[lo_] - lon0) * k * np.cos(np.deg2rad(lat0 * 1e-7))
        if self.heading_repr == "sincos":
            for msg, feat in HEADING_FEATURES.items():
                c = f"{msg}__{feat}"
                if c in row:
                    rad = np.deg2rad(row[c] / 100.0); row[f"{c}_sin"] = np.sin(rad); row[f"{c}_cos"] = np.cos(rad)
        return np.array([row[c] for c in self.cols], dtype=np.float32)


def replay_mission(telemetry_dir: str | Path, network_dir: str | Path, mission_id: str, enabled_messages: dict[str, bool],
                   freq_hz: float = 4.0, position_repr: str = "rel_start", heading_repr: str = "sincos") -> np.ndarray:
    """Feed the recorded CSVs of a mission in timestamp order through OnboardPreprocessor; (T, F) float32."""
    pp = OnboardPreprocessor(enabled_messages, freq_hz, position_repr, heading_repr)
    def tel_events(m):  # one generator per message type (a generator expression in the loop would bind m late)
        df = pd.read_csv(Path(telemetry_dir) / f"{mission_id}_{m}.csv")
        df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=TIMESTAMP_COL, keep="last")
        feats = MESSAGE_FEATURES[m]
        for ts, *vals in df[[TIMESTAMP_COL] + feats].itertuples(index=False):
            yield int(ts), 0, m, dict(zip(feats, vals))

    streams = [tel_events(m) for m in pp.tel]
    if pp.radio:
        df = pd.read_csv(Path(network_dir) / f"{mission_id}_{NETWORK_FILE}.csv")
        df = df.sort_values(TIMESTAMP_COL, kind="stable")
        streams.append(((int(ts), 1, "frame", (int(d), int(p), float(pl), int(sq)))
                        for ts, d, p, pl, sq in df[[TIMESTAMP_COL, "direction", "port", "payload_len", "seq_num"]].itertuples(index=False)))
    ends = []
    for m in pp.tel:
        ends.append(int(pd.read_csv(Path(telemetry_dir) / f"{mission_id}_{m}.csv", usecols=[TIMESTAMP_COL])[TIMESTAMP_COL].max()))
    if pp.radio:
        nd = pd.read_csv(Path(network_dir) / f"{mission_id}_{NETWORK_FILE}.csv", usecols=[TIMESTAMP_COL, "direction"])
        for s_ in pp.radio:
            sel = nd[(nd["direction"] == UPLINK_DIRECTION) if s_ in UPLINK_TYPES else (nd["direction"] != UPLINK_DIRECTION)]
            ends.append(int(sel[TIMESTAMP_COL].max()))
    pp.max_tick = min(ends)
    rows, last_ts = [], {}
    for ts, kind, name, val in heapq.merge(*streams, key=lambda e: (e[0], e[1])):
        if kind == 0:
            rows += pp.feed_message(name, ts, val); last_ts[name] = ts
        else:
            rows += pp.feed_frame(ts, *val)
            last_ts["radio_up" if val[0] == UPLINK_DIRECTION else "radio_down"] = ts
    rows += pp.flush(pp.max_tick)
    return np.stack(rows) if rows else np.zeros((0, len(pp.cols)), np.float32)
