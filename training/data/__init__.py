"""Data pipeline: feature definitions, preprocessing, windowing, datasets."""

from .features import MESSAGE_FEATURES, get_feature_columns, get_num_features
from .preprocessing import Normalizer, discover_missions, load_mission
from .windowing import build_window_index, compute_window_starts
from .dataset import (
    TelemetryWindowDataset,
    build_dataloaders,
    build_datasets,
    split_missions,
)

__all__ = [
    "MESSAGE_FEATURES",
    "get_feature_columns",
    "get_num_features",
    "Normalizer",
    "discover_missions",
    "load_mission",
    "build_window_index",
    "compute_window_starts",
    "TelemetryWindowDataset",
    "build_dataloaders",
    "build_datasets",
    "split_missions",
]