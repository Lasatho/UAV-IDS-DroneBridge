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
import time
from pathlib import Path

import torch

from data import build_dataloaders, get_num_features
from models import build_model
from config import load_config, set_seed

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Logging backends (wandb optional)
# ---------------------------------------------------------------------------

class MetricLogger:
    """Thin wrapper: wandb if configured+available, else stdout only."""

    def __init__(self, cfg: dict):
        self.backend = cfg["logging"]["backend"]
        self.run = None
        if self.backend == "wandb":
            try:
                import wandb
                self.run = wandb.init(
                    project=cfg["logging"]["project"], config=cfg
                )
            except Exception as e:  # offline, no login, not installed
                logger.warning("wandb unavailable (%s) — stdout only", e)
                self.backend = "none"

    def log(self, metrics: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------------------
# Train / validation passes
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, device, grad_clip: float,
    mlog: MetricLogger, global_step: int, log_every: int,
) -> tuple[float, int]:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model.training_step(batch)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.on_train_batch_end()

        total += loss.item() * batch.shape[0]
        n += batch.shape[0]
        global_step += 1
        if global_step % log_every == 0:
            mlog.log(
                {f"train/{k}": v.item() for k, v in out.items()},
                step=global_step,
            )
    return total / max(n, 1), global_step


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default=None)
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
    loaders, _ = build_dataloaders(cfg["data"])
    num_features = get_num_features(cfg["data"].get("features"))

    # --- Model ---
    name = cfg["model"]["name"]
    model = build_model(
        name,
        num_features=num_features,
        window_length=int(cfg["data"]["window_length"]),
        model_cfg=cfg["model"][name],
    ).to(device)

    if model.requires_labels:
        raise NotImplementedError(
            f"'{name}' is supervised and needs labeled attack windows. "
            "Label integration follows after WP1-WP4 (attacker node)."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s (%s trainable params)", name, f"{n_params:,}")

    tcfg = cfg["training"]
    optimizer, scheduler = model.configure_optimizers(tcfg)
    mlog = MetricLogger(cfg)

    # --- Checkpointing ---
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{name}"
    ckpt_dir = Path(tcfg["checkpoint_dir"]) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

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
    best_val = float("inf")
    epochs_no_improve = 0
    patience = int(tcfg["early_stopping_patience"])
    global_step = 0

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, loaders["train"], optimizer, device,
            float(tcfg["grad_clip_norm"]), mlog, global_step,
            int(cfg["logging"]["log_every_n_steps"]),
        )
        val_loss = validate(model, loaders["val"], device)
        if scheduler is not None:
            scheduler.step()

        mlog.log(
            {"epoch": epoch, "train/epoch_loss": train_loss,
             "val/loss": val_loss,
             "lr": optimizer.param_groups[0]["lr"]},
            step=global_step,
        )
        logger.info(
            "epoch %3d  train %.5f  val %.5f  (%.1fs)",
            epoch, train_loss, val_loss, time.time() - t0,
        )

        save("last", epoch, val_loss)
        if epoch % int(tcfg["save_every_n_epochs"]) == 0:
            save(f"epoch_{epoch:04d}", epoch, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            save("best", epoch, val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    logger.info("Best val loss: %.5f  (checkpoints: %s)", best_val, ckpt_dir)
    mlog.finish()


if __name__ == "__main__":
    main()