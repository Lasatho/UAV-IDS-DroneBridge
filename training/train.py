"""
Unified training entry point.

    python train.py --config configs/default.yaml \
                    [--experiment configs/exp_rssm.yaml] \
                    [overrides: model.name=rssm training.lr=3e-4 ...]

Model-agnostic: interacts with models exclusively via the
AnomalyDetectionModel interface. Validation metric for unsupervised
models is the validation loss; checkpointing keeps best + last.

Supervised models (requires_labels=True) need labeled windows, which do
not exist until attack data is integrated (WP1-WP4); the loop raises a
clear error in that case rather than silently training on garbage.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from data import build_dataloaders, get_num_features
from data.dataset import build_datasets
from data.stateful import StatefulChunkLoader
from models import build_model
from config import load_config, set_seed

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Logging backends (wandb optional)
# ---------------------------------------------------------------------------

def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) == 0


def _launch_tensorboard(logdir: str, port: int) -> None:
    """Best-effort background TensorBoard server; skips if one already serves the port."""
    if _port_is_open(port):
        logger.info("TensorBoard already running at http://localhost:%d", port)
        return
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info(
            "TensorBoard started: http://localhost:%d  (logdir=%s)", port, logdir
        )
    except FileNotFoundError:
        logger.warning(
            "tensorboard CLI not found on PATH — install via "
            "`pip install tensorboard` or start manually: "
            "tensorboard --logdir %s", logdir,
        )


class MetricLogger:
    """Thin wrapper: wandb or tensorboard if configured+available, else stdout only."""

    def __init__(self, cfg: dict, log_dir: Path | None = None):
        self.backend = cfg["logging"]["backend"]
        self.run = None
        self.writer = None
        if self.backend == "wandb":
            try:
                import wandb
                self.run = wandb.init(
                    project=cfg["logging"]["project"], config=cfg
                )
            except Exception as e:  # offline, no login, not installed
                logger.warning("wandb unavailable (%s) — stdout only", e)
                self.backend = "none"
        elif self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(log_dir))
                _launch_tensorboard(
                    cfg["training"]["checkpoint_dir"],
                    int(cfg["logging"].get("tensorboard_port", 6006)),
                )
            except Exception as e:  # not installed
                logger.warning("tensorboard unavailable (%s) — stdout only", e)
                self.backend = "none"

    def log(self, metrics: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)
        elif self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
        if self.writer is not None:
            self.writer.close()


# ---------------------------------------------------------------------------
# Train / validation passes
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, device, grad_clip: float,
    mlog: MetricLogger, global_step: int, log_every: int,
    max_stats: dict | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    spike: dict | None = None,
) -> tuple[float, int]:
    """One training epoch. If `max_stats` is given, it is filled with the
    per-epoch maximum of every training_step output (e.g. `kl_dyn`) and of
    the pre-clipping gradient norm (`grad_norm`) — diagnostics only, the
    training itself is unaffected.
    scaler (opt-in `training.amp`): mixed precision, training_step under fp16 autocast, loss scaling; gradients are
    unscaled before clipping, steps with inf/NaN gradients are skipped by the scaler (counted in the log).
    spike (opt-in training.spike_dump_threshold): dict thr, max, dir, n, epoch. A step whose loss exceeds thr is saved
    before the optimizer step (batch, RNG states before training_step, model weights, loss parts, group mask) so it
    can be replayed exactly; at most max files per run (Vault modellumbau_rssm_mtsjepa, 2026-11-14)."""
    model.train()
    total, n = 0.0, 0
    skipped = steps = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if spike is not None:
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            out = model.training_step(batch)
        loss = out["loss"]
        if spike is not None and spike["n"] < spike["max"] and not (loss.item() <= spike["thr"]):
            spike["dir"].mkdir(parents=True, exist_ok=True)
            gm = getattr(model, "last_gmask", None)
            torch.save({"epoch": spike["epoch"], "global_step": global_step, "batch": batch.detach().cpu(),
                        "rng_cpu": rng[0], "rng_cuda": rng[1], "gmask": gm.detach().cpu() if gm is not None else None,
                        "out": {k: v.detach().cpu() for k, v in out.items()},
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}},
                       spike["dir"] / ("spike_e%03d_s%07d.pt" % (spike["epoch"], global_step)))
            spike["n"] += 1
            logger.warning("spike: step %d loss %.4g > %.4g saved (%d of %d)", global_step, loss.item(), spike["thr"],
                           spike["n"], spike["max"])

        optimizer.zero_grad(set_to_none=True)
        grad_norm = None
        if scaler is None:
            loss.backward()
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            s0 = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < s0)
        steps += 1
        model.on_train_batch_end()

        if max_stats is not None:
            vals = {k: v.item() for k, v in out.items()}
            if grad_norm is not None:
                vals["grad_norm"] = grad_norm.item()
            for k, v in vals.items():
                cur = max_stats.get(k)
                # NaN stays sticky (max() would silently drop it)
                if cur is None or v != v or (cur == cur and v > cur):
                    max_stats[k] = v

        total += loss.item() * batch.shape[0]
        n += batch.shape[0]
        global_step += 1
        if global_step % log_every == 0:
            mlog.log(
                {f"train/{k}": v.item() for k, v in out.items()},
                step=global_step,
            )
    if scaler is not None:
        logger.info("amp: %d of %d steps skipped (inf/NaN gradients), loss scale %.0f", skipped, steps, scaler.get_scale())
    return total / max(n, 1), global_step


DRIFT_BINS = ((0, 0), (1, 4), (5, 19), (20, 79), (80, 10 ** 9))  # chunk index within the mission


def _lane_state(model, lanes, device):
    return (torch.zeros(lanes, model.deterministic_dim, device=device),
            torch.zeros(lanes, model.stochastic_dim, device=device),
            torch.zeros(lanes, model.num_features, device=device))


def run_stateful_epoch(model, loader, device, optimizer=None, grad_clip: float = 0.0,
                       mlog=None, global_step: int = 0, log_every: int = 1):
    """One pass over whole missions in chunks (opt-in training.stateful):
    training with optimizer, else validation. Also collects the state drift
    (mean |h| at the chunk end and KL_dyn per step, per bin of the chunk
    index within the mission)."""
    model.train(optimizer is not None)
    H, Z, P = _lane_state(model, loader.lanes, device)
    total, n = 0.0, 0
    drift = {b: [0.0, 0.0, 0] for b in DRIFT_BINS}
    with torch.set_grad_enabled(optimizer is not None):
        for x, reset, lane, pos in loader:
            x, reset, lane = x.to(device, non_blocking=True), reset.to(device), lane.to(device)
            out, (h1, z1), last = model.training_step_stateful(x, (H[lane], Z[lane]), P[lane], reset)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                model.on_train_batch_end()
                global_step += 1
                if mlog is not None and global_step % log_every == 0:
                    mlog.log({f"train/{k}": v.item() for k, v in out.items() if v.dim() == 0}, step=global_step)
            H[lane], Z[lane], P[lane] = h1, z1, last
            total += out["loss"].item() * x.shape[0]; n += x.shape[0]
            hn, kl = out["h_norm"].cpu().numpy(), out["kl_dyn_step"].cpu().numpy()
            for i, c in enumerate(pos.tolist()):
                for b in DRIFT_BINS:
                    if b[0] <= c <= b[1]:
                        d = drift[b]; d[0] += hn[i]; d[1] += kl[i]; d[2] += 1
                        break
    txt = "; ".join(f"chunk {b[0]}-{b[1] if b[1] < 10 ** 9 else 'end'}: |h| {d[0]/d[2]:.3f} kl {d[1]/d[2]:.3f} (n {d[2]})"
                    for b, d in drift.items() if d[2])
    return total / max(n, 1), global_step, txt


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model.training_step(batch)
        total += out["loss"].item() * batch.shape[0]
        n += batch.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def validate_norm(model, loader, device) -> dict:
    """MTS-JEPA (group tokens): token error per group relative to the spread of the targets.
    Error of a token = (1 - cos(pred, target)) * 2 / D; spread of a group = 1 - mean pairwise cosine of its
    L2-normalized target tokens over the validation set, from the mean vector m: (|m|^2 n - 1) / (n - 1).
    norm_loss_g = mean error_g * D / 2 / spread_g (~ 1 - R^2 in token space); val/norm_loss = mean over groups.
    The raw JEPA loss is smallest when the targets collapse (Vault modellumbau_rssm_mtsjepa, 2026-11-11), the
    normalized one is not."""
    model.eval()
    acc = None
    for batch in loader:
        st = model.val_stats(batch.to(device, non_blocking=True))
        acc = st if acc is None else {k: acc[k] + st[k] for k in acc}
    D = acc["t_sum"].shape[-1]; n = acc["t_n"]
    m = acc["t_sum"] / n.unsqueeze(-1)
    mean_cos = ((m ** 2).sum(-1) * n - 1) / (n - 1)
    spread = (1 - mean_cos).clamp(min=1e-6)
    err = acc["err_sum"] / acc["err_n"].clamp(min=1)
    norm = err * D / 2 / spread
    tg = acc["err_n"] > 0                                       # context-only groups have no hidden tokens
    out = {"val/norm_loss": float(norm[tg].mean())}
    names = getattr(model, "group_names", None) or [str(g) for g in range(len(norm))]
    for g, nm in enumerate(names):
        out[f"val_group/norm_{nm}"] = float(norm[g]); out[f"val_group/spread_{nm}"] = float(spread[g])
        out[f"val_group/err_{nm}"] = float(err[g])
    return out


def prepare_model_cfg(cfg, name, loaders, norm, stateful) -> None:
    """Data-dependent model settings, stored in the config so every checkpoint rebuilds the same model:
    channel groups (group masking / group tokens) and window_relative / loss weights. Shared with tune.py."""
    mcfg_ = cfg["model"][name]
    if (float(mcfg_.get("group_mask_prob", 0) or 0) > 0 or mcfg_.get("group_tokens")) and not mcfg_.get("groups"):
        # channel groups = MAVLink message / radio direction (prefix of the feature column)
        from data import get_feature_columns
        cols = get_feature_columns(cfg["data"].get("features"))
        names = list(dict.fromkeys(c.split("__")[0] for c in cols))
        cfg["model"][name]["groups"] = [[i for i, c in enumerate(cols) if c.split("__")[0] == g] for g in names]
        cfg["model"][name]["group_names"] = names
        logger.info("Channel groups for %s (%d): %s", name, len(names), names)
    _window_level_options(cfg, name, loaders, norm, stateful)
    if name == "rssm" and mcfg_.get("abs_inputs") and mcfg_.get("abs_input_idx") is None:
        # model.rssm.abs_inputs: names -> feature indices, stored in the config (checkpoint rebuilds the same model)
        from data import get_feature_columns
        cols = get_feature_columns(cfg["data"].get("features"))
        missing = [c for c in mcfg_["abs_inputs"] if c not in cols]
        if missing:
            raise ValueError(f"model.rssm.abs_inputs: unknown channels {missing}")
        mcfg_["abs_input_idx"] = [cols.index(c) for c in mcfg_["abs_inputs"]]
        logger.info("abs_inputs (encoder only, before window_relative): %s -> %s", mcfg_["abs_inputs"], mcfg_["abs_input_idx"])


def _window_level_options(cfg, name, loaders, norm, stateful) -> None:
    """data.window_relative and model.rssm.loss_weight: computed once on training windows and stored in the
    model config (so every checkpoint rebuilds the same transform / weights)."""
    mcfg = cfg["model"][name]
    wr = bool(cfg["data"].get("window_relative", False)) and not mcfg.get("window_relative")
    lw = name == "rssm" and str(mcfg.get("loss_weight", "none")) != "none" and not mcfg.get("loss_weights")
    inv = name == "mts_jepa" and mcfg.get("inv_noise_scale", "none") == "step" and mcfg.get("inv_noise_sd") is None
    if not (wr or lw or inv):
        return
    if stateful:
        raise NotImplementedError("window_relative / loss_weight / inv_noise_scale are implemented for windows only")
    from data import get_feature_columns
    from models.window_relative import build_spec, WindowRelative
    cols = get_feature_columns(cfg["data"].get("features"))
    ds = loaders["train"].dataset
    rng = np.random.default_rng(int(cfg["seed"]))
    W = np.stack([ds[int(i)].numpy() for i in rng.choice(len(ds), min(4000, len(ds)), replace=False)])
    if wr:
        spec = build_spec(cols, norm.mean, norm.std, W, cfg["data"].get("position_repr", "abs"),
                          exclude=cfg["data"].get("window_relative_exclude") or ())
        mcfg["window_relative"] = spec
        logger.info("window_relative: %s, spread within windows %s", spec["columns"], [round(v, 3) for v in spec["scale"]])
    x = torch.from_numpy(W)
    if mcfg.get("window_relative"):
        x = WindowRelative(mcfg["window_relative"])(x)
    sd = (x[:, 1:] - x[:, :-1]).reshape(-1, x.shape[-1]).std(0).numpy().astype(np.float64)  # step sd, model input space
    if inv:
        # model.mts_jepa.inv_noise_scale=step: invariance noise per channel in units of the sd of its step within a
        # window (around the mean step of the window, so a steady motion does not count as noise; as inv_scale.py),
        # mean over the windows
        sdw = (x[:, 1:] - x[:, :-1]).std(1).mean(0).numpy().astype(np.float64)
        sdi = np.where(sdw > 1e-6, sdw, 0.0)
        mcfg["inv_noise_sd"] = [float(v) for v in sdi]
        logger.info("inv_noise_scale step: step sd min %.3g, median %.3g, max %.3g (%s), %d constant channels without noise",
                    sdi[sdi > 0].min(), np.median(sdi[sdi > 0]), sdi.max(), cols[int(np.argmax(sdi))], int((sdi == 0).sum()))
    if lw:
        ok = sd > 1e-6
        w = np.ones_like(sd)
        w[ok] = 1.0 / sd[ok] ** (2 if mcfg["loss_weight"] == "step" else 1)
        fl = float(mcfg.get("loss_weight_floor", 0.0) or 0.0)
        if fl > 0:  # model.rssm.loss_weight_floor: lower bound as a fraction of the median weight (see models/rssm)
            lo = fl * np.median(w[ok]); nf = int((w[ok] < lo).sum()); w[ok] = np.maximum(w[ok], lo)
            logger.info("loss_weight_floor %.3g x median: %d channels raised", fl, nf)
        w[~ok] = np.median(w[ok])  # constant channels: typical weight
        w = w / w.mean()
        mcfg["loss_weights"] = [float(v) for v in w]
        logger.info("loss_weight %s: min %.3g, median %.3g, max %.3g (%s)", mcfg["loss_weight"], w.min(), np.median(w),
                    w.max(), cols[int(np.argmax(w))])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--stop-after-epochs", type=int, default=None,
                        help="stop after this many epochs, LR schedule over training.epochs "
                             "(same as a tune.py trial with --epochs)")
    parser.add_argument("overrides", nargs="*", help="a.b.c=value")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config, args.experiment, args.overrides)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Data ---
    tcfg = cfg["training"]
    stateful = bool(tcfg.get("stateful", False))
    if stateful:  # whole missions in chunks, state carried (RSSM only)
        dsets, norm = build_datasets(cfg["data"])
        C = int(tcfg.get("chunk_length") or cfg["data"]["window_length"])
        lanes = int(cfg["data"]["batch_size"])
        pool = int(tcfg.get("stateful_pool") or 0) or None  # opt-in: pool mode (full, decorrelated batches)
        loaders = {"train": StatefulChunkLoader(dsets["train"].missions, norm, C, lanes, True, int(cfg["seed"]), pool=pool),
                   "val": StatefulChunkLoader(dsets["val"].missions, norm, C, lanes, False)}
        logger.info("Stateful training: chunk %d, batch %d, pool %s, %d train / %d val chunks, %d train steps per epoch",
                    C, lanes, pool, loaders["train"].n_chunks, loaders["val"].n_chunks, len(loaders["train"]))
    else:
        loaders, norm = build_dataloaders(cfg["data"])
    num_features = get_num_features(cfg["data"].get("features"))

    # --- Model ---
    name = cfg["model"]["name"]
    prepare_model_cfg(cfg, name, loaders, norm, stateful)
    model = build_model(
        name,
        num_features=num_features,
        window_length=int(cfg["data"]["window_length"]),
        model_cfg=cfg["model"][name],
    ).to(device)

    if stateful and not hasattr(model, "training_step_stateful"):
        raise NotImplementedError(f"training.stateful needs a recurrent model with training_step_stateful, not '{name}'")
    if model.requires_labels:
        raise NotImplementedError(
            f"'{name}' is supervised and needs labeled attack windows. "
            "Label integration follows after WP1-WP4 (attacker node)."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s (%s trainable params)", name, f"{n_params:,}")

    optimizer, scheduler = model.configure_optimizers(tcfg)

    # --- Checkpointing ---
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{name}"
    ckpt_dir = Path(tcfg["checkpoint_dir"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    mlog = MetricLogger(cfg, log_dir=ckpt_dir)

    def save(tag: str, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg,
            },
            ckpt_dir / f"{tag}.pt",
        )

    # --- Loop ---
    select_metric = str(tcfg.get("select_metric", "loss"))
    if select_metric not in ("loss", "norm_loss", "composite"):
        raise ValueError(f"training.select_metric must be loss, norm_loss or composite, got {select_metric!r}")
    if select_metric == "norm_loss" and (stateful or not hasattr(model, "val_stats")):
        raise NotImplementedError("select_metric norm_loss needs a model with val_stats (MTS-JEPA group tokens), windows")
    # composite: the tuning objective J of tune_metric.py (normal val windows only, fixed disturbance catalogue) on
    # training.metric_windows fixed val windows (seed 0, as tune.py --objective composite) after every epoch
    # amp (opt-in): mixed precision for the training steps (fp16 autocast + loss scaling); validation, composite
    # metric and evaluations stay in fp32. Measured 2026-11-12: ~1.7x faster, gradients cos >= 0.998 (Vault 9.29)
    scaler = None
    if bool(tcfg.get("amp", False)):
        if device.type != "cuda" or stateful:
            raise NotImplementedError("training.amp needs a CUDA device and window training (not stateful)")
        scaler = torch.amp.GradScaler("cuda")
        logger.info("amp: fp16 autocast + GradScaler for the training steps")
    W_metric = None
    if select_metric == "composite":
        if stateful:
            raise NotImplementedError("select_metric composite is implemented for windows only")
        import tune_metric
        from data import get_feature_columns
        W_metric = tune_metric.sample_windows(loaders["val"].dataset, int(tcfg.get("metric_windows", 256)), seed=0)
        metric_columns = get_feature_columns(cfg["data"].get("features"))
    best_val = float("inf")
    epochs_no_improve = 0
    patience = int(tcfg["early_stopping_patience"])
    # early stopping counts an epoch as improvement only if the metric drops by more than this (absolute, default 0);
    # best.pt is still the plain minimum
    min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
    best_ref = float("inf")
    global_step = 0

    # training.spike_log (opt-in): per-epoch maximum of every training_step output and of the gradient norm in the log;
    # training.spike_dump_threshold (opt-in, > 0): save steps with a loss above it (train_one_epoch, spike)
    spike_log = bool(tcfg.get("spike_log", False))
    spike = None
    if float(tcfg.get("spike_dump_threshold", 0) or 0) > 0:
        if stateful:
            raise NotImplementedError("training.spike_dump_threshold is implemented for windows only")
        spike = {"thr": float(tcfg["spike_dump_threshold"]), "max": int(tcfg.get("spike_dump_max", 3)),
                 "dir": ckpt_dir / "spikes", "n": 0, "epoch": 0}
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        t0 = time.time()
        if spike is not None:
            spike["epoch"] = epoch
        max_stats = {} if spike_log else None
        if stateful:
            train_loss, global_step, _ = run_stateful_epoch(
                model, loaders["train"], device, optimizer, float(tcfg["grad_clip_norm"]),
                mlog, global_step, int(cfg["logging"]["log_every_n_steps"]))
            val_loss, _, drift = run_stateful_epoch(model, loaders["val"], device)
            logger.info("state drift val (mean |h| at chunk end, KL_dyn per step): %s", drift)
        else:
            train_loss, global_step = train_one_epoch(
                model, loaders["train"], optimizer, device,
                float(tcfg["grad_clip_norm"]), mlog, global_step,
                int(cfg["logging"]["log_every_n_steps"]), scaler=scaler, max_stats=max_stats, spike=spike,
            )
            if max_stats:
                logger.info("train max in epoch %d: %s", epoch, {k: float(f"{v:.5g}") for k, v in max_stats.items()})
            val_loss = validate(model, loaders["val"], device)
        val_extra = {}
        if select_metric == "norm_loss":
            val_extra = validate_norm(model, loaders["val"], device)
            logger.info("val norm_loss %.5f, per group %s", val_extra["val/norm_loss"],
                        {k.split("/")[-1]: round(v, 4) for k, v in val_extra.items() if k.startswith("val_group/norm")})
        if select_metric == "composite":
            tm0 = time.time()
            # training.composite_terms (default "abc"): terms of J; "ab" = prediction measures only (see tune_metric)
            J, parts = tune_metric.composite(model, name, W_metric, metric_columns, device,
                                             terms=str(tcfg.get("composite_terms", "abc")))
            val_extra = {f"val_metric/{k}": float(parts[k]) for k in ("J", "a", "b", "c", "auc_mean", "r2_mean") if k in parts}
            logger.info("val composite J %.5f (a %.4f, b %.4f, c %s; %.0fs)", J, parts["a"], parts["b"],
                        f"{parts['c']:.4f}" if "c" in parts else "-", time.time() - tm0)
        val_metric = (val_extra["val/norm_loss"] if select_metric == "norm_loss"
                      else val_extra["val_metric/J"] if select_metric == "composite" else val_loss)
        if scheduler is not None:
            scheduler.step()

        mlog.log(
            {"epoch": epoch, "train/epoch_loss": train_loss,
             "val/loss": val_loss,
             "lr": optimizer.param_groups[0]["lr"], **val_extra},
            step=global_step,
        )
        logger.info(
            "epoch %3d  train %.5f  val %.5f  (%.1fs)",
            epoch, train_loss, val_loss, time.time() - t0,
        )

        save("last", epoch, val_loss)
        if epoch % int(tcfg["save_every_n_epochs"]) == 0:
            save(f"epoch_{epoch:04d}", epoch, val_loss)

        if val_metric < best_val:
            best_val = val_metric
            save("best", epoch, val_loss)
        if val_metric < best_ref - min_delta:
            best_ref = val_metric
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break
        if args.stop_after_epochs and epoch >= args.stop_after_epochs:
            logger.info("Stopped after epoch %d (--stop-after-epochs)", epoch)
            break

    logger.info("Best val %s: %.5f  (checkpoints: %s)", {"loss": "loss", "composite": "composite J"}.get(select_metric, select_metric),
                best_val, ckpt_dir)
    mlog.finish()


if __name__ == "__main__":
    main()