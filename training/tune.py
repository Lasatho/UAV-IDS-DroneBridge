"""
Optuna hyperparameter search for the unsupervised anomaly detectors
(rssm, mts_jepa).

Objective: validation loss (best value seen across the trial's epochs).
This is a proxy for anomaly-detection quality, not the metric itself —
switch to AUC-PR on labeled windows once phase_labels -> telemetry
timebase mapping lands (see evaluate.py:load_window_labels).

Search space is joint per model (shared + architecture-specific
hyperparameters sampled together, not staged) — see SEARCH_SPACES
below for what is tuned vs. held fixed at its configs/default.yaml
value, and why.

    python train.py ...   # unaffected; this script does not modify it

Usage:
    cd training
    python tune.py --model rssm      --n-trials 50 --epochs 30
    python tune.py --model mts_jepa  --n-trials 50 --epochs 30 \
        --experiment configs/flypaw_test.yaml

Results: optuna_studies/<model>.db (SQLite). Inspect with:
    optuna-dashboard sqlite:///optuna_studies/rssm.db
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import optuna
import torch

from config import load_config, set_seed
from data import build_dataloaders, get_num_features
from models import build_model
from train import MetricLogger, train_one_epoch, validate

logger = logging.getLogger("tune")


# ---------------------------------------------------------------------------
# Search spaces: joint per model. Everything else in configs/default.yaml
# (batch_size, window_length/stride, scheduler, warmup_epochs,
# grad_clip_norm, and each model's remaining fields) stays fixed at its
# default — see the report printed at the end of main() for the exact
# fixed values used in a given run.
# ---------------------------------------------------------------------------

def _rssm_space(trial: optuna.Trial) -> dict:
    return {
        "training": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "rssm": {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
                "deterministic_dim": trial.suggest_categorical("deterministic_dim", [64, 128, 256]),
                "kl_dyn_beta": trial.suggest_float("kl_dyn_beta", 0.1, 2.0, log=True),
                "free_nats": trial.suggest_float("free_nats", 0.5, 3.0),
            }
        },
    }


def _mts_jepa_space(trial: optuna.Trial) -> dict:
    return {
        "training": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "mts_jepa": {
                "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 256]),
                "depth": trial.suggest_categorical("depth", [2, 4, 6]),
                "predictor_depth": trial.suggest_categorical("predictor_depth", [1, 2, 3]),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.3, 0.7),
                # must divide data.window_length (fixed, see module docstring)
                "patch_length": trial.suggest_categorical("patch_length", [4, 8, 16]),
            }
        },
    }


SEARCH_SPACES = {"rssm": _rssm_space, "mts_jepa": _mts_jepa_space}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(dict(out.get(k, {})), v) if isinstance(v, dict) else v
    return out


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    model_name: str, base_cfg: dict, loaders: dict, num_features: int,
    device: torch.device, epochs: int,
):
    quiet_logger = MetricLogger({"logging": {"backend": "none"}})

    def objective(trial: optuna.Trial) -> float:
        cfg = _merge(base_cfg, SEARCH_SPACES[model_name](trial))
        set_seed(int(cfg["seed"]))

        model = build_model(
            model_name,
            num_features=num_features,
            window_length=int(cfg["data"]["window_length"]),
            model_cfg=cfg["model"][model_name],
        ).to(device)
        optimizer, scheduler = model.configure_optimizers(cfg["training"])

        best_val = float("inf")
        global_step = 0
        for epoch in range(1, epochs + 1):
            _, global_step = train_one_epoch(
                model, loaders["train"], optimizer, device,
                float(cfg["training"]["grad_clip_norm"]), quiet_logger,
                global_step, int(cfg["logging"]["log_every_n_steps"]),
            )
            val_loss = validate(model, loaders["val"], device)
            if scheduler is not None:
                scheduler.step()
            best_val = min(best_val, val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30,
                         help="Epochs per trial (< full training run; pruning cuts bad trials earlier)")
    parser.add_argument("--storage-dir", default="optuna_studies")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config, args.experiment, [])
    cfg["training"]["epochs"] = args.epochs  # keep cosine schedule matched to per-trial budget

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # window_length/stride, resample_freq_hz and features are fixed across
    # all trials -> dataloaders (and the underlying .npy cache) are built
    # once and reused, not rebuilt per trial.
    loaders, _ = build_dataloaders(cfg["data"])
    num_features = get_num_features(cfg["data"].get("features"))

    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_dir / args.model}.db"

    study = optuna.create_study(
        study_name=args.model,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["seed"])),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    objective = make_objective(args.model, cfg, loaders, num_features, device, args.epochs)
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best val loss: %.5f", study.best_value)
    logger.info("Best params: %s", study.best_params)
    logger.info("Study storage: %s (open with `optuna-dashboard %s`)", storage, storage)


if __name__ == "__main__":
    main()
