"""
Configuration handling.

- Loads a base YAML config (configs/default.yaml).
- Optionally merges an experiment YAML on top (deep merge).
- Applies CLI overrides in dot notation: training.lr=3e-4 model.name=rssm
- Resolves ${a.b.c} interpolation references.
- Global seeding for reproducibility.
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_INTERP = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _get_by_path(cfg: dict, path: str) -> Any:
    node: Any = cfg
    for key in path.split("."):
        node = node[key]
    return node


def _set_by_path(cfg: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    """Parse a CLI override value via YAML (handles int/float/bool/list).

    YAML 1.1 parses '3e-4' as string (requires '3.0e-4'); fall back to
    float conversion for such cases.
    """
    val = yaml.safe_load(raw)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def _resolve_interpolations(cfg: dict) -> dict:
    """Resolve ${a.b.c} references in string values (single pass per level,
    repeated until fixpoint; cycles raise)."""
    cfg = copy.deepcopy(cfg)

    def resolve_str(s: str) -> Any:
        m = _INTERP.fullmatch(s)
        if m:  # whole-string reference: keep referenced type
            return _get_by_path(cfg, m.group(1))
        return _INTERP.sub(lambda m: str(_get_by_path(cfg, m.group(1))), s)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and "${" in node:
            return resolve_str(node)
        return node

    for _ in range(10):  # fixpoint iteration, bounded
        resolved = walk(cfg)
        if resolved == cfg:
            return resolved
        cfg = resolved
    raise ValueError("Unresolvable (cyclic?) config interpolation")


def load_config(
    base_path: str | Path,
    experiment_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Load base config, merge experiment config and CLI overrides.

    Args:
        base_path: Path to default.yaml.
        experiment_path: Optional experiment YAML merged on top.
        overrides: List of "a.b.c=value" strings (highest precedence).
    """
    cfg = yaml.safe_load(Path(base_path).read_text())

    if experiment_path:
        exp = yaml.safe_load(Path(experiment_path).read_text()) or {}
        cfg = _deep_merge(cfg, exp)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        path, raw = item.split("=", 1)
        _set_by_path(cfg, path.strip(), _parse_value(raw))

    return _resolve_interpolations(cfg)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch; optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False