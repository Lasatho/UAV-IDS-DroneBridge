"""
Optuna hyperparameter search for the unsupervised anomaly detectors
(rssm, mts_jepa).

Objective: mean anomaly score on the validation split after the trial's
last epoch, i.e. the quantity the IDS thresholds — not the training loss.
The last epoch and not the best one: a configuration that diverges after
an early optimum must not be ranked by its pre-divergence value. The training loss is unsuitable here because tuned
hyperparameters change its formula (RSSM: kl_dyn_beta/free_nats set a hard
lower bound) or its scale (MTS-JEPA: MSE over L2-normalized embeddings
scales with 1/embed_dim). MTS-JEPA scores are converted to 1 - cos so they
are independent of embed_dim. Still a proxy for detection quality; the best
configurations are ranked afterwards by AUC-PR on labeled attack missions.

Trials use the learning-rate schedule of a full training run (warmup +
cosine over training.epochs from the config) and stop after --epochs
epochs. A schedule compressed to the trial length would anneal the LR to
~0 within the trial, hide instabilities that appear at high LR and tune
the LR for a regime the full run never sees.

Diagnostics stored per trial (user attrs, not part of the ranking):
per-epoch maxima of the training_step outputs and the gradient norm
(RSSM KL stability), and for MTS-JEPA the effective rank of the target
embeddings (representation collapse check).

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

Seed replicates of the best configurations of a finished study (separate
study <source>_seeds, no pruning):
    python tune.py --model rssm --replicate-from rssm_score1b --top 5 --seeds 1 2

Results: optuna_studies/<study>.db (SQLite, default study <model>_score1b).
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
            # Upper bounds from stage 1: lr > ~2e-3 or kl_dyn_beta > ~0.6
            # led to KL explosions; the best configurations sat at the edge
            # of the former architecture ranges (hidden 128, deterministic 256).
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "rssm": {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "deterministic_dim": trial.suggest_categorical("deterministic_dim", [128, 256, 512]),
                "kl_dyn_beta": trial.suggest_float("kl_dyn_beta", 0.1, 0.8, log=True),
                "free_nats": trial.suggest_float("free_nats", 0.5, 3.0),
            }
        },
    }


def _mts_jepa_space(trial: optuna.Trial) -> dict:
    return {
        "training": {
            # Stage 1: best lr 1.3e-4..9.5e-4 (lower edge opened), depth 6
            # (upper edge, opened to 8), predictor_depth 1 (3 never competitive).
            "lr": trial.suggest_float("lr", 5e-5, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "mts_jepa": {
                "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 256]),
                "depth": trial.suggest_categorical("depth", [4, 6, 8]),
                "predictor_depth": trial.suggest_categorical("predictor_depth", [1, 2]),
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
    """Effective rank of the (L2-normalized) target embeddings.

    A low prediction error is only meaningful if the embeddings carry
    information; near-constant embeddings are trivially predictable.
    emb_eff_rank: exp(entropy of the normalized singular values) of the
    centered token matrix (Roy & Vetterli 2007), between 1 and embed_dim.
    (The per-component std is not reported: for unit vectors it is ~1/sqrt(D)
    regardless of the information content.)
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
    return {"emb_eff_rank": eff_rank}


def make_objective(
    model_name: str, base_cfg: dict, loaders: dict, num_features: int,
    device: torch.device, epochs: int, replicate: bool = False,
):
    quiet_logger = MetricLogger({"logging": {"backend": "none"}})

    def objective(trial: optuna.Trial) -> float:
        if replicate and "seed" not in trial.user_attrs:
            # Queue drained by another worker: the sampler proposed a new
            # configuration, which does not belong in a replicate study.
            trial.set_user_attr("not_a_replicate", True)
            raise optuna.TrialPruned()
        cfg = _merge(base_cfg, SEARCH_SPACES[model_name](trial))
        # Seed replicates (--replicate-from) carry their own seed.
        seed = int(trial.user_attrs.get("seed", cfg["seed"]))
        trial.set_user_attr("seed", seed)
        set_seed(seed)

        model = build_model(
            model_name,
            num_features=num_features,
            window_length=int(cfg["data"]["window_length"]),
            model_cfg=cfg["model"][model_name],
        ).to(device)
        optimizer, scheduler = model.configure_optimizers(cfg["training"])

        score = float("nan")
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

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if model_name == "mts_jepa":
            for k, v in jepa_embedding_stats(model, loaders["val"], device).items():
                trial.set_user_attr(k, v)

        return score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _create_study(study_name: str, storage, cfg: dict, worker_id: int,
                  pruning: bool = True) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(cfg["seed"]) + worker_id),
        pruner=(optuna.pruners.MedianPruner(n_warmup_steps=5) if pruning
                else optuna.pruners.NopPruner()),
    )


def _enqueue_replicates(study: optuna.Study, source: optuna.Study,
                        top: int, seeds: list[int]) -> None:
    """Queue the `top` best completed configurations of `source` once per seed.

    Idempotent: (params, seed) pairs already in `study` are skipped, so
    parallel workers or restarts do not duplicate work.
    """
    # Waiting (enqueued, not yet started) trials hold their parameters in
    # the "fixed_params" system attribute, not in .params.
    done = {(tuple(sorted((t.params or t.system_attrs.get("fixed_params", {})).items())),
             t.user_attrs.get("seed"))
            for t in study.trials}
    best = sorted(
        (t for t in source.trials if t.state == optuna.trial.TrialState.COMPLETE),
        key=lambda t: t.value,
    )[:top]
    for t in best:
        for s in seeds:
            if (tuple(sorted(t.params.items())), s) in done:
                continue
            study.enqueue_trial(t.params, user_attrs={"seed": s,
                                                      "source_trial": t.number})


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
                        help="Default <model>_score1b (stage 1b; earlier studies "
                             "are kept as <model>.db and <model>_score.db)")
    parser.add_argument("--replicate-from", default=None, metavar="STUDY",
                        help="Re-train the --top best configurations of STUDY "
                             "with each of --seeds (study STUDY_seeds, no pruning)")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Distinct per parallel process on the same study "
                             "(offsets the TPE sampler seed)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config, args.experiment, [])
    # training.epochs is NOT overridden: the LR schedule stays that of a full
    # run, the trial just stops after --epochs epochs (see module docstring).
    if args.epochs > int(cfg["training"]["epochs"]):
        parser.error("--epochs exceeds training.epochs of the config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # window_length/stride, resample_freq_hz and features are fixed across
    # all trials -> dataloaders (and the underlying .npy cache) are built
    # once and reused, not rebuilt per trial.
    loaders, _ = build_dataloaders(cfg["data"])
    num_features = get_num_features(cfg["data"].get("features"))

    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    def _storage(name: str) -> optuna.storages.RDBStorage:
        return optuna.storages.RDBStorage(
            f"sqlite:///{storage_dir / name}.db",
            # parallel workers write to the same SQLite file
            engine_kwargs={"connect_args": {"timeout": 60}},
        )

    replicate = args.replicate_from is not None
    if replicate:
        study_name = args.study_name or f"{args.replicate_from}_seeds"
    else:
        study_name = args.study_name or f"{args.model}_score1b"
    storage = _storage(study_name)

    # Parallel workers may race on study creation; the loser just loads it.
    try:
        study = _create_study(study_name, storage, cfg, args.worker_id,
                              pruning=not replicate)
    except optuna.exceptions.DuplicatedStudyError:
        study = _create_study(study_name, storage, cfg, args.worker_id,
                              pruning=not replicate)

    n_trials = args.n_trials
    if replicate:
        source = optuna.load_study(study_name=args.replicate_from,
                                   storage=_storage(args.replicate_from))
        _enqueue_replicates(study, source, args.top, args.seeds)
        n_trials = sum(t.state == optuna.trial.TrialState.WAITING
                       for t in study.trials)
        logger.info("Replicates queued: %d", n_trials)

    def _stop_when_queue_empty(study: optuna.Study, _trial) -> None:
        if replicate and not any(t.state == optuna.trial.TrialState.WAITING
                                 for t in study.trials):
            study.stop()

    objective = make_objective(args.model, cfg, loaders, num_features, device,
                               args.epochs, replicate=replicate)
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[_stop_when_queue_empty])

    logger.info("Best val score: %.6g", study.best_value)
    logger.info("Best params: %s", study.best_params)
    logger.info("Study storage: %s", storage.url)


if __name__ == "__main__":
    main()
