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
        self, cfg: dict
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