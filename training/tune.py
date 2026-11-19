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

--objective composite (decided 2026-11-11, Vault modellumbau_rssm_mtsjepa): instead of the mean
anomaly score, J = (a + b + c) / 3 from tune_metric.py on fixed normal validation windows (normalized
prediction error, information content of the representation, AUC over a generic disturbance catalogue);
the raw scores and losses are unsuitable (MTS-JEPA: smallest when the targets collapse).

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

After widening the search space, continue in a new study that starts from
the finished trials of the old one (one process per worker as above):
    python tune.py --model rssm --study-name rssm_v2c --warm-start-from rssm_v2b \
        --gpu-mem-fraction 0.19 --worker-id 0 ...

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
from train import MetricLogger, train_one_epoch, prepare_model_cfg
import tune_metric
from data import get_feature_columns

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
            # Study rssm_v2b (cleaned data): best at deterministic 1024 and
            # free_nats ~2.9 (upper edges); the stage-1 limits were never
            # checked against the GPS placeholder artefact. All upper bounds
            # opened one step to find where the range really ends.
            # Study rssm_v2r (raster): best 5 all at deterministic 2048, 4 of 5 at
            # hidden 512 (upper edges): deterministic 3072/4096 and hidden
            # 768/1024 added (study rssm_v2r_ext with warm start; benchmark
            # 2026-11-08: 4096/1024 peaks at 2.5 GiB, ~3x the step time of 2048/512).
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "rssm": {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512, 768, 1024]),
                "deterministic_dim": trial.suggest_categorical("deterministic_dim", [128, 256, 512, 1024, 2048, 3072, 4096]),
                "kl_dyn_beta": trial.suggest_float("kl_dyn_beta", 0.1, 1.5, log=True),
                "free_nats": trial.suggest_float("free_nats", 0.5, 6.0),
            }
        },
    }


def _mts_jepa_space(trial: optuna.Trial) -> dict:
    return {
        "training": {
            # Stage 1: best lr 1.3e-4..9.5e-4 (lower edge opened), depth 6
            # (upper edge, opened to 8), predictor_depth 1 (3 never competitive).
            # Study mts_jepa_v2: best 5 all at depth 10 (upper edge, opened to 14).
            # Study mts_jepa_v2b: best 5 all at depth 14 (upper edge, opened to
            # 20, sampled as an ordered integer); lr upper bound opened to 5e-3.
            # Study mts_jepa_v2c: best at depth 20 (upper edge, opened to 28).
            # Study mts_jepa_v2r: best 5 all at embed_dim 256, 2 of 5 at depth 28
            # (upper edges): embed_dim 384 added, depth opened to 36. 512 does not
            # fit next to the running rssm_v2r workers (memory test 2026-11-07).
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "mts_jepa": {
                "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 256, 384]),
                "depth": trial.suggest_int("depth", 4, 36, step=2),
                "predictor_depth": trial.suggest_categorical("predictor_depth", [1, 2]),
            }
        },
    }


def _rssm_space_v3(trial: optuna.Trial) -> dict:
    """Tuning v3 (decision Lasse 2026-11-10, Vault modellumbau_rssm_mtsjepa "Entscheidungen vor dem Tuning"):
    sizes not beyond the previous edges (Jetson; the rssm_v2r_ext values 3072/4096 and 768/1024 were never run
    and are dropped), new: share of group masking. Structural choices (position_repr, window_relative,
    hidden_base, loss_weight, predict_increment) are fixed per study with --set."""
    return {
        "training": {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "rssm": {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
                "deterministic_dim": trial.suggest_categorical("deterministic_dim", [512, 1024, 2048]),
                "kl_dyn_beta": trial.suggest_float("kl_dyn_beta", 0.1, 1.5, log=True),
                "free_nats": trial.suggest_float("free_nats", 0.5, 6.0),
                "group_mask_prob": trial.suggest_float("group_mask_prob", 0.1, 0.5),
            }
        },
    }


def _mts_jepa_space_v3(trial: optuna.Trial) -> dict:
    """Tuning v3 (decision 2026-11-10): sizes as before (embed_dim up to 384, depth up to 36), new: shares of the
    mask types temporal / group / causal (three weights in [0.1, 1], normalized to sum 1) and the maximum causal
    length (1-4 of 8 patches). Needs group tokens (--set model.mts_jepa.group_tokens=true); target_attn,
    window_relative, vc_weight fixed per study with --set."""
    w = [trial.suggest_float(f"mask_w_{k}", 0.1, 1.0) for k in ("temporal", "group", "causal")]
    return {
        "training": {
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "model": {
            "mts_jepa": {
                "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 256, 384]),
                "depth": trial.suggest_int("depth", 4, 36, step=2),
                "predictor_depth": trial.suggest_categorical("predictor_depth", [1, 2]),
                "mask_mix": [v / sum(w) for v in w],
                "causal_max": trial.suggest_int("causal_max", 1, 4),
            }
        },
    }


SEARCH_SPACES = {"rssm": _rssm_space, "mts_jepa": _mts_jepa_space}
SEARCH_SPACES_V3 = {"rssm": _rssm_space_v3, "mts_jepa": _mts_jepa_space_v3}


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
    norm=None, objective_kind: str = "score", metric_windows: int = 256, spaces: dict | None = None,
    early_stop_patience: int = 0,
):
    # early_stop_patience (composite only, 0 = off): stop a trial after this many epochs without a lower J and
    # return its best J (best epoch in the user attr "best_epoch"); decided 2026-11-12 for MTS-JEPA, whose J has a
    # clear optimum after a few epochs (Vault training_runs_4hz 9.29)
    if early_stop_patience and objective_kind != "composite":
        raise ValueError("early_stop_patience needs the composite objective")
    spaces = spaces or SEARCH_SPACES
    quiet_logger = MetricLogger({"logging": {"backend": "none"}})
    # composite objective (tune_metric.py): fixed normal validation windows, the same for every trial
    W_metric = (tune_metric.sample_windows(loaders["val"].dataset, metric_windows, seed=0)
                if objective_kind == "composite" else None)
    columns = get_feature_columns(base_cfg["data"].get("features"))

    def objective(trial: optuna.Trial) -> float:
        # A configuration too large for the worker's GPU share is a failed
        # trial (NaN -> TrialState.FAIL), not a crash of the worker.
        try:
            return _objective(trial)
        except torch.cuda.OutOfMemoryError:
            trial.set_user_attr("oom", True)
            torch.cuda.empty_cache()
            return float("nan")

    def _objective(trial: optuna.Trial) -> float:
        if replicate and "seed" not in trial.user_attrs:
            # Queue drained by another worker: the sampler proposed a new
            # configuration, which does not belong in a replicate study.
            trial.set_user_attr("not_a_replicate", True)
            raise optuna.TrialPruned()
        cfg = _merge(base_cfg, spaces[model_name](trial))
        if spaces is SEARCH_SPACES_V3 and model_name == "mts_jepa" and not cfg["model"]["mts_jepa"].get("group_tokens"):
            raise ValueError("search space v3 for mts_jepa needs --set model.mts_jepa.group_tokens=true")
        prepare_model_cfg(cfg, model_name, loaders, norm, False)
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
        # training.amp (opt-in, --set training.amp=true): mixed precision for the training steps
        scaler = torch.amp.GradScaler("cuda") if cfg["training"].get("amp") else None
        best_J, best_epoch, no_improve = float("inf"), 0, 0
        for epoch in range(1, epochs + 1):
            max_stats: dict = {}
            train_loss, global_step = train_one_epoch(
                model, loaders["train"], optimizer, device,
                float(cfg["training"]["grad_clip_norm"]), quiet_logger,
                global_step, int(cfg["logging"]["log_every_n_steps"]),
                max_stats=max_stats, scaler=scaler,
            )
            epoch_max.append(max_stats)
            trial.set_user_attr("epoch_max", epoch_max)
            if objective_kind == "composite":
                score, parts = tune_metric.composite(model, model_name, W_metric, columns, device)
                trial.set_user_attr("metric_epochs", trial.user_attrs.get("metric_epochs", []) +
                                    [{k: parts[k] for k in ("J", "a", "b", "c", "auc_mean", "r2_mean")}])
                trial.set_user_attr("metric_last", parts)
            else:
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
            if early_stop_patience:
                if score < best_J:
                    best_J, best_epoch, no_improve = score, epoch, 0
                else:
                    no_improve += 1
                trial.set_user_attr("best_epoch", best_epoch)
                if no_improve >= early_stop_patience:
                    trial.set_user_attr("early_stopped_at_epoch", epoch)
                    break

        if model_name == "mts_jepa" and not cfg["model"]["mts_jepa"].get("group_tokens"):
            for k, v in jepa_embedding_stats(model, loaders["val"], device).items():
                trial.set_user_attr(k, v)

        return best_J if early_stop_patience else score

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


class _DistributionRecorder:
    """Stands in for a trial: answers each suggest_* call with the value of
    `params` and records the distribution the search space now uses."""

    def __init__(self, params: dict):
        self.params, self.distributions = params, {}

    def _record(self, name, dist):
        self.distributions[name] = dist
        return self.params[name]

    def suggest_float(self, name, low, high, *, step=None, log=False):
        return self._record(name, optuna.distributions.FloatDistribution(low, high, step=step, log=log))

    def suggest_int(self, name, low, high, *, step=1, log=False):
        return self._record(name, optuna.distributions.IntDistribution(low, high, step=step, log=log))

    def suggest_categorical(self, name, choices):
        return self._record(name, optuna.distributions.CategoricalDistribution(choices))


def _warm_start(study: optuna.Study, source: optuna.Study, model_name: str) -> None:
    """Copy the finished trials of `source` into `study` under the current
    search space, so a widened space does not discard what TPE has learned.

    Optuna refuses changed categorical choices within one study, hence a new
    study. Idempotent: trials already copied (user attr warm_start_from) are
    skipped, so every worker can call it at start. Trials whose values lie
    outside the current space are not copied.
    """
    done = {t.user_attrs.get("warm_start_from") for t in study.trials}
    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL)
    n = 0
    for t in source.trials:
        tag = f"{source.study_name}#{t.number}"
        if t.state not in finished or tag in done or not t.params:
            continue
        rec = _DistributionRecorder(t.params)
        SEARCH_SPACES[model_name](rec)
        if not all(d._contains(d.to_internal_repr(t.params[k]))
                   for k, d in rec.distributions.items()):
            logger.warning("Warm start: %s outside the current space, skipped", tag)
            continue
        # FrozenTrial instead of create_trial: keeps the start/end times, so
        # durations per trial stay meaningful in the new study
        study.add_trial(optuna.trial.FrozenTrial(
            number=-1, trial_id=-1, state=t.state,
            value=t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
            datetime_start=t.datetime_start, datetime_complete=t.datetime_complete,
            params=t.params, distributions=rec.distributions,
            user_attrs={**t.user_attrs, "warm_start_from": tag},
            system_attrs={}, intermediate_values=t.intermediate_values,
        ))
        n += 1
    logger.info("Warm start: %d trials copied from %s", n, source.study_name)


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
    parser.add_argument("--warm-start-from", default=None, metavar="STUDY",
                        help="Copy the finished trials of STUDY into the study "
                             "under the current search space (after widening it)")
    parser.add_argument("--gpu-mem-fraction", type=float, default=None,
                        help="Cap this process's share of GPU memory, so an "
                             "oversized trial fails alone instead of starving "
                             "parallel workers")
    parser.add_argument("--objective", choices=["score", "composite"], default="score",
                        help="score: mean anomaly score on val (stage 1/2 studies); composite: tune_metric.py, "
                             "normal data + generic disturbance catalogue (decided 2026-11-11)")
    parser.add_argument("--metric-windows", type=int, default=256,
                        help="normal val windows for the composite objective")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="composite only: stop a trial after N epochs without a lower J, return its best J (0 = off)")
    parser.add_argument("--space", choices=["v2", "v3"], default="v2",
                        help="v2: search spaces of the studies up to rssm_v2r/mts_jepa_v2r_ext; v3: tuning v3 "
                             "(decision 2026-11-10: sizes not extended, group masking / mask mixture)")
    parser.add_argument("--set", nargs="*", default=[], metavar="A.B=C",
                        help="config overrides (as train.py), applied before the search space")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Distinct per parallel process on the same study "
                             "(offsets the TPE sampler seed)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config, args.experiment, args.set)
    # training.epochs is NOT overridden: the LR schedule stays that of a full
    # run, the trial just stops after --epochs epochs (see module docstring).
    if args.epochs > int(cfg["training"]["epochs"]):
        parser.error("--epochs exceeds training.epochs of the config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if args.gpu_mem_fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)

    # window_length/stride, resample_freq_hz and features are fixed across
    # all trials -> dataloaders (and the underlying .npy cache) are built
    # once and reused, not rebuilt per trial.
    loaders, norm = build_dataloaders(cfg["data"])
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

    if args.warm_start_from:
        _warm_start(study, optuna.load_study(study_name=args.warm_start_from,
                                             storage=_storage(args.warm_start_from)),
                    args.model)

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
                               args.epochs, replicate=replicate, norm=norm,
                               objective_kind=args.objective, metric_windows=args.metric_windows,
                               spaces=SEARCH_SPACES_V3 if args.space == "v3" else SEARCH_SPACES,
                               early_stop_patience=args.early_stop_patience)
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[_stop_when_queue_empty])

    logger.info("Best val score: %.6g", study.best_value)
    logger.info("Best params: %s", study.best_params)
    logger.info("Study storage: %s", storage.url)


if __name__ == "__main__":
    main()
