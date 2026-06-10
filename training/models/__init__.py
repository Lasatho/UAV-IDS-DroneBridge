"""Model registry: instantiate any architecture from config."""

from .base import AnomalyDetectionModel
from .rssm.model import RSSM
from .mts_jepa.model import MTSJEPA
from .hassler.model import HasslerBaseline

_REGISTRY: dict[str, type[AnomalyDetectionModel]] = {
    "rssm": RSSM,
    "mts_jepa": MTSJEPA,
    "hassler": HasslerBaseline,
}


def build_model(
    name: str, num_features: int, window_length: int, model_cfg: dict
) -> AnomalyDetectionModel:
    """Instantiate a model by name with its config sub-section.

    Args:
        name: Key in the registry ("rssm" | "mts_jepa" | "hassler").
        num_features: Input feature count (from data pipeline).
        window_length: Timesteps per window (must match data config).
        model_cfg: kwargs for the architecture (cfg["model"][name]).
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](
        num_features=num_features, window_length=window_length, **model_cfg
    )


__all__ = ["AnomalyDetectionModel", "RSSM", "MTSJEPA", "HasslerBaseline", "build_model"]