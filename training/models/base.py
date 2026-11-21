"""
Common interface for all anomaly-detection models in the comparative study.

Every architecture (RSSM, MTS-JEPA, supervised baseline) implements:

    training_step(batch)  -> scalar loss          (architecture-specific objective)
    anomaly_score(batch)  -> (B,) float tensor    (higher = more anomalous)
    configure_optimizers(cfg) -> (optimizer, scheduler | None)

The unified training loop and the evaluation code interact with models
exclusively through this interface; no architecture-specific branching
outside the model packages.

Batch layout: float32 tensor (B, T, F) as produced by the data pipeline.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AnomalyDetectionModel(nn.Module, ABC):
    """Base class for all models in the comparative study."""

    def __init__(self, num_features: int, window_length: int):
        super().__init__()
        self.num_features = num_features
        self.window_length = window_length

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the training objective for one batch.

        Returns a dict with at least key "loss" (scalar, differentiable).
        Additional entries (e.g. loss components) are logged as metrics.
        """

    @abstractmethod
    @torch.no_grad()
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        """Per-window anomaly score, shape (B,). Higher = more anomalous.

        This is the quantity thresholded by the IDS and compared across
        architectures in the evaluation.
        """

    def configure_optimizers(
        self, cfg: dict, steps_per_epoch: int | None = None
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        """Default: AdamW + optional cosine schedule, with linear LR warmup.

        `warmup_epochs` ramps the LR linearly from near-zero to the
        configured value before the main schedule takes over. Without it,
        Transformer-based models (MTS-JEPA) see the full LR from step 1,
        which destabilizes attention/LayerNorm weights early on and shows
        up as steadily diverging train/val loss rather than convergence.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        epochs = int(cfg["epochs"])

        # training.warmup_unit "step" (default): warmup and cosine per optimizer step; the scheduler gets the
        # attribute per_step = True and the caller steps it after every optimizer step. "epoch": the former
        # schedule, stepped once per epoch, so with warmup_epochs 1 the whole first epoch ran at start_factor x lr
        # and epoch 2 started with a 1000x jump (Vault modellumbau_rssm_mtsjepa, 2026-11-20).
        unit = str(cfg.get("warmup_unit", "step"))
        if unit not in ("step", "epoch"):
            raise ValueError(f"training.warmup_unit must be step or epoch, got {unit!r}")
        if unit == "step" and (warmup_epochs > 0 or cfg.get("scheduler") == "cosine"):
            if not steps_per_epoch:
                raise ValueError("training.warmup_unit 'step' needs steps_per_epoch (or set warmup_unit epoch)")
            w = warmup_epochs * int(steps_per_epoch)
            total = max(epochs * int(steps_per_epoch), w + 1)
            cosine = cfg.get("scheduler") == "cosine"

            def factor(step: int) -> float:
                if step < w:
                    return 1e-3 + (1.0 - 1e-3) * step / w
                if not cosine:
                    return 1.0
                return 0.5 * (1.0 + math.cos(math.pi * min(step - w, total - w) / (total - w)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
            scheduler.per_step = True
            return optimizer, scheduler

        main_scheduler = None
        if cfg.get("scheduler") == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup_epochs, 1)
            )

        if warmup_epochs <= 0:
            return optimizer, main_scheduler

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs
        )
        if main_scheduler is None:
            return optimizer, warmup_scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Hooks with no-op defaults
    # ------------------------------------------------------------------

    def on_train_batch_end(self) -> None:
        """Called after every optimizer step (e.g. EMA updates)."""

    @property
    def requires_labels(self) -> bool:
        """True for supervised models (training needs attack labels)."""
        return False