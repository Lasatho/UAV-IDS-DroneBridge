"""
MTS-JEPA: Joint Embedding Predictive Architecture for multivariate
time series.

JEPA principle applied to temporal patches:

    1. Window (B, T, F) is split into non-overlapping temporal patches,
       each linearly embedded to a token -> (B, N, D).
    2. A subset of patches is masked. The context encoder (Transformer)
       sees only visible patches.
    3. A predictor forecasts the embeddings of masked patches from the
       context, conditioned on their positions.
    4. Targets are produced by an EMA copy of the encoder over the FULL
       window (stop-gradient).

Loss = MSE between predicted and target embeddings of masked patches
(prediction in embedding space — no input-space reconstruction).

Anomaly score = same embedding prediction error at eval time with a
fixed deterministic mask pattern (reproducible scores).

Reference points: Assran et al. 2023 (I-JEPA, masking + EMA target),
MTS-JEPA 2026 (temporal patching for multivariate series). Exact
hyperparameters follow the configs, not hard-coded paper values.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import AnomalyDetectionModel


class _TransformerEncoder(nn.Module):
    """Thin wrapper: pre-norm Transformer encoder over token sequences."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


def _sincos_positional(n: int, dim: int) -> torch.Tensor:
    """Fixed sin-cos positional embeddings, shape (1, n, dim)."""
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim // 2, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * 2 * i / dim)
    angles = pos * freq  # (n, dim/2)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return emb.unsqueeze(0)


class MTSJEPA(AnomalyDetectionModel):
    def __init__(
        self,
        num_features: int,
        window_length: int,
        patch_length: int = 8,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        predictor_depth: int = 2,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
    ):
        super().__init__(num_features, window_length)
        assert window_length % patch_length == 0, \
            "window_length must be divisible by patch_length"
        assert embed_dim % 2 == 0, \
            "embed_dim must be even (sin/cos positional encoding splits it in half)"
        self.patch_length = patch_length
        self.num_patches = window_length // patch_length
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        n_masked = int(round(self.num_patches * mask_ratio))
        assert 0 < n_masked < self.num_patches, "mask_ratio yields 0 or all patches"
        self.n_masked = n_masked

        patch_dim = patch_length * num_features
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.register_buffer(
            "pos_embed", _sincos_positional(self.num_patches, embed_dim)
        )

        self.context_encoder = _TransformerEncoder(embed_dim, depth, num_heads)
        self.predictor = _TransformerEncoder(embed_dim, predictor_depth, num_heads)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # EMA target branch (no gradients)
        self.target_patch_embed = nn.Linear(patch_dim, embed_dim)
        self.target_encoder = _TransformerEncoder(embed_dim, depth, num_heads)
        self._init_target()

        # Deterministic eval mask: n_masked patches spread evenly over the
        # window (generalizes the fixed "every other patch" pattern to the
        # configured mask_ratio, instead of hard-coding 50%).
        eval_mask = torch.zeros(self.num_patches, dtype=torch.bool)
        eval_idx = (torch.arange(self.n_masked) * self.num_patches) // self.n_masked
        eval_mask[eval_idx] = True
        self.register_buffer("eval_mask", eval_mask)

    # ------------------------------------------------------------------
    # EMA handling
    # ------------------------------------------------------------------

    def _target_modules(self):
        return [
            (self.patch_embed, self.target_patch_embed),
            (self.context_encoder, self.target_encoder),
        ]

    def _init_target(self):
        for online, target in self._target_modules():
            target.load_state_dict(online.state_dict())
            for p in target.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _ema_update(self):
        d = self.ema_decay
        for online, target in self._target_modules():
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.mul_(d).add_(p_o.detach(), alpha=1 - d)

    def on_train_batch_end(self) -> None:
        self._ema_update()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, N, patch_length*F)."""
        B, T, Feat = x.shape
        x = x.reshape(B, self.num_patches, self.patch_length * Feat)
        return x

    def _random_mask(self, B: int, device) -> torch.Tensor:
        """Boolean mask (B, N), True = masked. Exactly n_masked per row."""
        scores = torch.rand(B, self.num_patches, device=device)
        idx = scores.argsort(dim=1)[:, : self.n_masked]
        mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask

    def _forward_jepa(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted vs. target embeddings of masked patches.

        Returns (pred, target), both (B, n_masked_per_row, D) flattened
        over masked positions: (n_total_masked, D).
        """
        B = x.shape[0]
        patches = self._patchify(x)
        tokens = self.patch_embed(patches) + self.pos_embed  # (B, N, D)

        # Context: replace masked tokens by mask token (keeps positions —
        # simpler than dropping, equivalent information for the predictor).
        mask_tok = self.mask_token + self.pos_embed  # broadcast (1, N, D)
        ctx_in = torch.where(mask.unsqueeze(-1), mask_tok.expand(B, -1, -1), tokens)
        ctx = self.context_encoder(ctx_in)
        pred = self.predictor(ctx)  # (B, N, D)

        with torch.no_grad():
            tgt_tokens = self.target_patch_embed(patches) + self.pos_embed
            target = self.target_encoder(tgt_tokens)  # (B, N, D)

        return pred[mask], target[mask]

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = self._random_mask(batch.shape[0], batch.device)
        pred, target = self._forward_jepa(batch, mask)
        loss = F.mse_loss(pred, target)
        return {"loss": loss}

    @torch.no_grad()
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        self.eval()
        B = batch.shape[0]
        mask = self.eval_mask.unsqueeze(0).expand(B, -1)
        patches = self._patchify(batch)
        tokens = self.patch_embed(patches) + self.pos_embed
        mask_tok = self.mask_token + self.pos_embed
        ctx_in = torch.where(mask.unsqueeze(-1), mask_tok.expand(B, -1, -1), tokens)
        pred = self.predictor(self.context_encoder(ctx_in))
        tgt = self.target_encoder(self.target_patch_embed(patches) + self.pos_embed)
        err = ((pred - tgt) ** 2).mean(dim=-1)  # (B, N)
        # Mean prediction error over masked positions only
        return (err * mask).sum(dim=1) / mask.sum(dim=1)