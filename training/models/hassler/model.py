"""
Supervised baseline after Hassler, Mughal & Ismail (IEEE TITS 2024,
DOI: 10.1109/TITS.2023.3340272).

STATUS: ARCHITECTURE SKELETON — the layer configuration below is a
generic CNN-LSTM placeholder. Before any reported results, this module
must be aligned with the exact architecture from the paper (layer
sizes, kernel widths, pooling, classifier head). Marked with TODOs.

Training is supervised: requires window-level attack labels, which the
unsupervised models do not use. The training loop passes (x, y) batches
to this model (see requires_labels).

Anomaly score = attack-class probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import AnomalyDetectionModel


class HasslerBaseline(AnomalyDetectionModel):
    # TODO: verify against Hassler et al. 2024 before producing results:
    #   - conv channel progression / kernel sizes
    #   - LSTM hidden size and depth
    #   - classifier head layout, dropout placement
    #   - their input feature engineering (if any) vs. our raw windows

    def __init__(
        self,
        num_features: int,
        window_length: int,
        conv_channels: tuple[int, ...] = (64, 128),
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__(num_features, window_length)

        convs = []
        in_ch = num_features
        for out_ch in conv_channels:
            convs += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),  # binary: normal vs. attack
        )

    @property
    def requires_labels(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> logits (B,)."""
        z = self.conv(x.transpose(1, 2))      # (B, C, T)
        z, _ = self.lstm(z.transpose(1, 2))   # (B, T, H)
        return self.head(z[:, -1]).squeeze(-1)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        x, y = batch  # supervised: (windows, labels)
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        return {"loss": loss}

    @torch.no_grad()
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        self.eval()
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return torch.sigmoid(self(x))