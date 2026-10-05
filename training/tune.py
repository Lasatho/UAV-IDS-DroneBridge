"""
Optuna hyperparameter search for the unsupervised anomaly detectors
(rssm, mts_jepa).

Objective: mean anomaly score on the validation split (best value seen
across the trial's epochs), i.e. the quantity the IDS thresholds — not the
training loss. The training loss is unsuitable here because tuned
hyperparameters change its formula (RSSM: kl_dyn_beta/free_nats set a hard
lower bound) or its scale (MTS-JEPA: MSE over L2-normalized embeddings
scales with 1/embed_dim). MTS-JEPA scores are converted to 1 - cos so they
are independent of embed_dim. Still a proxy for detection quality; the best
configurations are ranked afterwards by AUC-PR on labeled attack missions.

Diagnostics stored per trial (user attrs, not part of the ranking):
per-epoch maxima of the training_step outputs and the gradient norm
(RSSM KL stability), and for MTS-JEPA the embedding spread / effective
rank of the target embeddings (representation collapse check).

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

Parallel workers on one GPU (the RSSM step loop is Python-bound, the GPU is
mostly idle): start several processes on the same study, each with its own
--worker-id so their TPE samplers do not propose identical trials:
    python tune.py --model rssm --n-trials 7 --epochs 12 --worker-id 0 &
    python tune.py --model rssm --n-trials 7 --epochs 12 --worker-id 1 &
--n-trials counts per process.

Results: optuna_studies/<study>.db (SQLite, default study <model>_score).
Inspect with:
    optuna-dashboard sqlite:///optuna_studies/rssm_score.db
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import optuna
import torch
import torch.nn.functional as F

from config import load_config, set_seed
from data import build_dataloaders, get_num_features
from models import build_model
from train import MetricLogger, train_one_epoch

logger = logging.getLogger("tune")


# ---------------------------------------------------------------------------
# Search spaces: joint per model. Everything else in configs/default.yaml
# (batch_size, window_length/stride, scheduler, warmup_epochs,
# grad_clip_norm, and each model's remaining fields) stays fixed at its
# default. MTS-JEPA patch_length/mask_ratio are fixed there on purpose:
# they define the prediction task itself (see default.yaml).
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

@torch.no_grad()
def val_score(model, model_name: str, loader, device) -> float:
    """Mean anomaly score over the validation split.

    MTS-JEPA scores are mean squared differences of L2-normalized
    embeddings, averaged over embed_dim components: score = (2 - 2 cos) / D.
    Multiplying by D/2 gives 1 - cos, independent of embed_dim.
    """
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        total += model.anomaly_score(batch).sum().item()
        n += batch.shape[0]
    score = total / max(n, 1)
    if model_name == "mts_jepa":
        score *= model.patch_embed.out_features / 2
    return score


@torch.no_grad()
def jepa_embedding_stats(model, loader, device, max_batches: int = 50) -> dict:
    """Spread and effective rank of the (L2-normalized) target embeddings.

    A low prediction error is only meaningful if the embeddings carry
    information; near-constant embeddings are trivially predictable.
    emb_std: mean per-component std across tokens (0 = collapsed).
    emb_eff_rank: exp(entropy of the normalized singular values) of the
    centered token matrix (Roy & Vetterli 2007), between 1 and embed_dim.
    """
    model.eval()
    embs = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        patches = model._patchify(batch)
        tgt = model.target_encoder(model.target_patch_embed(patches) + model.pos_embed)
        embs.append(F.normalize(tgt, dim=-1).reshape(-1, tgt.shape[-1]).float())
    z = torch.cat(embs)
    z = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(z)
    p = sv / sv.sum()
    eff_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    return {"emb_std": z.std(dim=0).mean().item(), "emb_eff_rank": eff_rank}


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

        best_score = float("inf")
        global_step = 0
        epoch_max: list[dict] = []
        for epoch in range(1, epochs + 1):
            max_stats: dict = {}
            train_loss, global_step = train_one_epoch(
                model, loaders["train"], optimizer, device,
                float(cfg["training"]["grad_clip_norm"]), quiet_logger,
                global_step, int(cfg["logging"]["log_every_n_steps"]),
                max_stats=max_stats,
            )
            epoch_max.append(max_stats)
            trial.set_user_attr("epoch_max", epoch_max)
            score = val_score(model, model_name, loaders["val"], device)
            trial.set_user_attr("last_train_loss", train_loss)
            if scheduler is not None:
                scheduler.step()

            # Diverged run: report as failed (NaN return -> TrialState.FAIL)
            # instead of ranking it with a meaningless value.
            if not (math.isfinite(score) and math.isfinite(train_loss)):
                trial.set_user_attr("diverged_at_epoch", epoch)
                return float("nan")

            best_score = min(best_score, score)
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if model_name == "mts_jepa":
            for k, v in jepa_embedding_stats(model, loaders["val"], device).items():
                trial.set_user_attr(k, v)

        return best_score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _create_study(study_name: str, storage, cfg: dict, worker_id: int) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["seed"]) + worker_id),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30,
                         help="Epochs per trial (< full training run; pruning cuts bad trials earlier)")
    parser.add_argument("--storage-dir", default="optuna_studies")
    parser.add_argument("--study-name", default=None,
                        help="Default <model>_score (score-based objective; the "
                             "earlier loss-based studies are kept as <model>.db)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Distinct per parallel process on the same study "
                             "(offsets the TPE sampler seed)")
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
    study_name = args.study_name or f"{args.model}_score"
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{storage_dir / study_name}.db",
        # parallel workers write to the same SQLite file
        engine_kwargs={"connect_args": {"timeout": 60}},
    )

    # Parallel workers may race on study creation; the loser just loads it.
    try:
        study = _create_study(study_name, storage, cfg, args.worker_id)
    except optuna.exceptions.DuplicatedStudyError:
        study = _create_study(study_name, storage, cfg, args.worker_id)

    objective = make_objective(args.model, cfg, loaders, num_features, device, args.epochs)
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best val score: %.6g", study.best_value)
    logger.info("Best params: %s", study.best_params)
    logger.info("Study storage: %s", storage.url)


if __name__ == "__main__":
    main()
