"""
PyTorch Dataset and DataLoader factory for UAV telemetry windows.

Pipeline:
    discover_missions -> mission-level split -> load+align (cached as .npy)
    -> fit Normalizer on train -> sliding-window index -> Dataset

Mission-level splitting guarantees that no mission contributes windows
to more than one split (no temporal leakage).

Caching: aligned missions are stored as float32 .npy under
`<dataset_dir>/cache/<freq>hz/`. CSV parsing therefore happens once per
mission and resample frequency; subsequent runs memory-map the arrays.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import get_feature_columns
from .preprocessing import Normalizer, discover_missions, load_mission
from .windowing import build_window_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_missions(
    mission_ids: list[str],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministic mission-level train/val/test split."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(mission_ids))
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))

    return {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train : n_train + n_val].tolist(),
        "test": ids[n_train + n_val :].tolist(),
    }


# ---------------------------------------------------------------------------
# Cached mission loading
# ---------------------------------------------------------------------------

def _cache_dir(dataset_dir: Path, resample_freq_hz: float, feature_cols: list[str]) -> Path:
    """Cache directory keyed by frequency and feature set."""
    feat_hash = hashlib.sha1("|".join(feature_cols).encode()).hexdigest()[:8]
    return Path(dataset_dir) / "cache" / f"{resample_freq_hz:g}hz_{feat_hash}"


def load_mission_cached(
    telemetry_dir: Path,
    cache_dir: Path,
    mission_id: str,
    resample_freq_hz: float,
    enabled_messages: dict[str, bool] | None,
) -> np.ndarray:
    """Aligned feature array (T, n_features) for one mission, via .npy cache."""
    cache_path = cache_dir / f"{mission_id}.npy"
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    df = load_mission(
        telemetry_dir,
        mission_id,
        resample_freq_hz=resample_freq_hz,
        enabled_messages=enabled_messages,
    )
    arr = df.drop(columns=["t_ns"]).to_numpy(dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return np.load(cache_path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TelemetryWindowDataset(Dataset):
    """Sliding windows over normalized telemetry of one split.

    __getitem__ returns a float32 tensor of shape
    (window_length, n_features). Models that need (B, C, T) layouts
    transpose internally.
    """

    def __init__(
        self,
        missions: list[np.ndarray],
        mission_ids: list[str],
        window_length: int,
        stride: int,
        normalizer: Normalizer,
    ):
        assert len(missions) == len(mission_ids)
        self.missions = missions
        self.mission_ids = mission_ids
        self.window_length = window_length
        self.normalizer = normalizer
        self.index = build_window_index(
            [m.shape[0] for m in missions], window_length, stride
        )

    def __len__(self) -> int:
        return self.index.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        m_idx, start = self.index[i]
        window = self.missions[m_idx][start : start + self.window_length]
        window = self.normalizer(np.asarray(window, dtype=np.float32))
        return torch.from_numpy(window)

    def window_meta(self, i: int) -> tuple[str, int]:
        """(mission_id, start) of window i — for evaluation traceability."""
        m_idx, start = self.index[i]
        return self.mission_ids[m_idx], int(start)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict) -> tuple[dict[str, TelemetryWindowDataset], Normalizer]:
    """Construct train/val/test datasets from the data config section.

    Expects keys: dataset_dir, telemetry_dir, resample_freq_hz,
    window_length, window_stride, split_ratios, split_seed, features,
    plus optional max_missions (debug subsampling).

    Side effects: writes split assignment and normalizer stats to
    `<dataset_dir>/cache/` for reproducibility.

    Cache-only mode: if telemetry_dir has no mission CSVs but a complete
    cache (splits.json + normalizer.json + one .npy per mission) exists
    for this frequency/feature combination, missions are loaded straight
    from the cache. This is the deploy path — the cache directory alone
    (no raw CSVs) is a sufficient, portable training artifact.
    """
    dataset_dir = Path(cfg["dataset_dir"])
    telemetry_dir = Path(cfg["telemetry_dir"])
    enabled = cfg.get("features")
    feature_cols = get_feature_columns(enabled)
    freq = float(cfg["resample_freq_hz"])
    cache_dir = _cache_dir(dataset_dir, freq, feature_cols)
    logger.info("Cache directory: %s", cache_dir)

    splits_path = cache_dir / "splits.json"
    norm_path = cache_dir / "normalizer.json"
    have_csvs = telemetry_dir.exists() and any(telemetry_dir.glob("mission_*_*.csv"))
    cache_only = not have_csvs and splits_path.exists() and norm_path.exists()

    if cache_only:
        splits = json.loads(splits_path.read_text())
        missing = [
            mid
            for ids in splits.values()
            for mid in ids
            if not (cache_dir / f"{mid}.npy").exists()
        ]
        if missing:
            raise RuntimeError(
                f"Cache-only load requested ({telemetry_dir} has no CSVs) "
                f"but cache is missing .npy for: {missing}"
            )
        n_missions = sum(len(ids) for ids in splits.values())
        logger.info(
            "No telemetry CSVs in %s — loading %d missions from cache-only artifact %s",
            telemetry_dir, n_missions, cache_dir,
        )
    else:
        mission_ids = discover_missions(telemetry_dir)
        if not mission_ids:
            raise RuntimeError(f"No complete missions found in {telemetry_dir}")

        max_missions = cfg.get("max_missions")
        if max_missions:
            mission_ids = mission_ids[: int(max_missions)]
        logger.info("Using %d missions", len(mission_ids))

        splits = split_missions(
            mission_ids,
            ratios=tuple(cfg["split_ratios"]),
            seed=int(cfg["split_seed"]),
        )

    # Load (with cache) per split; skip missions that fail alignment.
    arrays: dict[str, list[np.ndarray]] = {}
    ids_used: dict[str, list[str]] = {}
    for split, ids in splits.items():
        arrs, used = [], []
        for mid in ids:
            try:
                arrs.append(
                    load_mission_cached(telemetry_dir, cache_dir, mid, freq, enabled)
                )
                used.append(mid)
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Skipping %s: %s", mid, e)
        arrays[split], ids_used[split] = arrs, used

    if not arrays["train"]:
        raise RuntimeError("Training split is empty after loading")

    # Normalizer: fit on train only, persist next to the cache.
    if norm_path.exists():
        normalizer = Normalizer.load(norm_path)
        if normalizer.columns != feature_cols:
            normalizer = Normalizer.fit(arrays["train"], feature_cols)
            normalizer.save(norm_path)
    else:
        normalizer = Normalizer.fit(arrays["train"], feature_cols)
        normalizer.save(norm_path)

    (cache_dir / "splits.json").write_text(json.dumps(ids_used, indent=2))

    datasets = {
        split: TelemetryWindowDataset(
            missions=arrays[split],
            mission_ids=ids_used[split],
            window_length=int(cfg["window_length"]),
            stride=int(cfg["window_stride"]),
            normalizer=normalizer,
        )
        for split in ("train", "val", "test")
    }
    for split, ds in datasets.items():
        logger.info("%s: %d missions, %d windows", split, len(ds.missions), len(ds))
    return datasets, normalizer


def build_dataloaders(
    cfg: dict,
) -> tuple[dict[str, DataLoader], Normalizer]:
    """DataLoaders for all splits. Shuffling only on train."""
    datasets, normalizer = build_datasets(cfg)
    loaders = {
        split: DataLoader(
            ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=bool(cfg.get("pin_memory", False)),
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }
    return loaders, normalizer