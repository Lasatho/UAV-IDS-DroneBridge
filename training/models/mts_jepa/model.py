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

Group tokens (opt-in `group_tokens`, decided 2026-11-10, Vault
modellumbau_rssm_mtsjepa): instead of one token per patch over all channels,
one token per patch AND channel group (MAVLink message / radio direction,
`groups` = channel indices per group), N x G tokens with a temporal and a
group position embedding. Attention can then compare e.g. the GPS and the
EKF token of the same time directly. Training masks are drawn per window
from a mixture `mask_mix` = (temporal, group, causal) probabilities:
temporal = the usual random patches (all groups), group = one random group
over the whole window (the model must infer it from the other groups:
sensor redundancy), causal = the last 1..causal_max patches (prediction of
the newest data from the past, as online). The eval mask stays the fixed
temporal pattern; token_errors() gives the error per patch and group for
any mask (group scores, group masks at evaluation).

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
from ..window_relative import WindowRelative


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(self.encoder(x, mask=mask))


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
        group_tokens: bool = False,
        groups: list[list[int]] | None = None,
        group_names: list[str] | None = None,
        mask_mix: list[float] | None = None,
        causal_max: int = 2,
        target_attn: str = "full",
        window_relative: dict | None = None,
        vc_weight: float = 0.0,
        vc_cov_ratio: float = 0.04,
        vc_gamma: float = 1.0,
        vc_no_dropout: bool = False,
        vc_space: str = "raw",
        target_no_dropout: bool = False,
        inv_weight: float = 0.0,
        inv_noise: float = 0.05,
        inv_noise_scale: str = "none",
        inv_noise_sd: list[float] | None = None,
        recon_weight: float = 0.0,
        group_mask_max: int = 1,
        context_only_groups: list[str] | None = None,
        fourier_weight: float = 0.0,
        fourier_dim: int = 32,
        fourier_scale: float = 1.0,
        fourier_encoder: bool = True,
        fourier_seed: int = 0,
        predict_var: bool = False,
        var_weight: float = 1.0,
        self_head: bool = False,
        self_weight: float = 1.0,
        self_context: int = 2,
        score_mode: str = "mse",
        group_mask_sets: list[list[str]] | None = None,
        group_mask_set_prob: float = 0.0,
        group_mask_start_max: int = 0,
        consistency_sets: list[list[str]] | None = None,
        group_mask_set_mode: str = "replace",
        ac: bool = False,
        ac_action_group: str = "SERVO_OUTPUT_RAW",
        ac_radio_groups: list[str] | None = None,
        ac_horizon: int = 2,
        ac_group_mask_prob: float = 0.3,
        ac_score: str = "token",
        ac_recon_norm: dict | None = None,
        ac_use_action: bool = True,
        ac_one_way_radio: bool = True,
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

        self.group_tokens = bool(group_tokens)
        if self.group_tokens:
            if not groups:
                raise ValueError("group_tokens needs groups (channel indices per group)")
            self.groups = [list(map(int, g)) for g in groups]
            self.group_names = list(group_names) if group_names else None
            self.n_groups = len(self.groups)
            mix = list(mask_mix) if mask_mix is not None else [0.5, 0.3, 0.2]
            assert len(mix) == 3 and abs(sum(mix) - 1.0) < 1e-6, "mask_mix = (temporal, group, causal), sum 1"
            self.register_buffer("mask_mix", torch.tensor(mix, dtype=torch.float32))
            self.causal_max = int(causal_max)
            assert 1 <= self.causal_max < self.num_patches
            for gi, idx in enumerate(self.groups):
                self.register_buffer(f"gidx_{gi}", torch.tensor(idx, dtype=torch.long), persistent=False)
            self.group_embed = nn.ModuleList(nn.Linear(patch_length * len(g), embed_dim) for g in self.groups)
            self.target_group_embed = nn.ModuleList(nn.Linear(patch_length * len(g), embed_dim) for g in self.groups)
            pe = _sincos_positional(self.num_patches, embed_dim)  # (1, N, D)
            self.register_buffer("gpos_embed", pe.unsqueeze(2))   # (1, N, 1, D)
            self.group_pos = nn.Parameter(torch.zeros(1, 1, self.n_groups, embed_dim))
            nn.init.normal_(self.group_pos, std=0.02)
            for p_ in self.target_group_embed.parameters():
                p_.requires_grad_(False)
            self.target_group_embed.load_state_dict(self.group_embed.state_dict())
            # Attention of the target encoder: "full" = all N x G tokens (as the context
            # encoder), "group" = each token only sees the tokens of its own group (over
            # time). With "full" the target tokens of poorly predictable groups (radio,
            # servos) were found to carry mainly content of other groups, so their own
            # channels hardly reach the target. "group" makes each target token a function
            # of its own channels only.
            if target_attn not in ("full", "group"):
                raise ValueError(f"target_attn must be 'full' or 'group', got {target_attn!r}")
            self.target_attn = target_attn
            gid = torch.arange(self.num_patches * self.n_groups) % self.n_groups  # token order (N, G) flattened
            self.register_buffer("target_attn_mask", gid.view(-1, 1) != gid.view(1, -1), persistent=False)
        elif target_attn != "full":
            raise ValueError("target_attn 'group' needs group_tokens")
        if window_relative and not self.group_tokens:
            raise ValueError("window_relative is implemented for group_tokens only")
        # window_relative: level channels relative to the window start (models/window_relative.py)
        self.win_rel = WindowRelative(window_relative) if window_relative else None
        # vc_weight (opt-in, default 0): variance/covariance term against representation collapse, as in
        # VICReg (Bardes et al. 2022), on the online counterpart of the target tokens (context encoder on the
        # unmasked window, same attention as the target encoder; the target encoder is its EMA). Per patch
        # position and group over the batch, so the time/group position embedding cannot supply the spread:
        # var = mean_d relu(gamma - std_d), cov = sum of squared off-diagonal covariances / D. Training only;
        # the validation loss stays the plain JEPA loss. Measured collapse without it: servo/GPS target tokens
        # nearly constant at epoch 4 (Vault modellumbau_rssm_mtsjepa, 2026-11-11).
        if vc_weight < 0 or vc_cov_ratio < 0 or vc_gamma <= 0:
            raise ValueError("vc_weight, vc_cov_ratio >= 0 and vc_gamma > 0 required")
        if vc_weight > 0 and not self.group_tokens:
            raise ValueError("vc_weight is implemented for group_tokens only")
        self.vc_weight, self.vc_cov_ratio, self.vc_gamma = float(vc_weight), float(vc_cov_ratio), float(vc_gamma)
        # vc_no_dropout (opt-in): the VC pass runs the context encoder without dropout (gradient kept). With
        # dropout the model met the variance hinge through amplified dropout noise instead of differences between
        # windows: mts_wr_vc epoch 15 var 0.024 in train mode, 0.817 in eval mode on the same windows
        # (Vault modellumbau_rssm_mtsjepa, 2026-11-13).
        if vc_no_dropout and self.vc_weight <= 0:
            raise ValueError("vc_no_dropout needs vc_weight > 0")
        self.vc_no_dropout = bool(vc_no_dropout)
        # vc_space (opt-in): "raw" = VC term on the raw online tokens (default). "normalized" = on sqrt(D) * the
        # L2-normalized tokens, the space of the JEPA loss and the invariance term. In raw space a common offset of
        # all tokens leaves std and covariance unchanged but bundles the normalized tokens (C long: offset 0.43 ->
        # 1.25, mean pairwise cosine 0.20 -> 0.61 from epoch 4; Vault modellumbau_rssm_mtsjepa, 2026-11-14). With
        # unit-norm tokens the per-dim variances sum to 1 - |mean|^2, so the scaled hinge (std 1 = isotropic,
        # centered) penalizes the offset directly.
        if vc_space not in ("raw", "normalized"):
            raise ValueError(f"vc_space must be raw or normalized, not {vc_space!r}")
        if vc_space != "raw" and self.vc_weight <= 0:
            raise ValueError("vc_space normalized needs vc_weight > 0")
        self.vc_space = vc_space
        # inv_weight (opt-in, default 0): noise invariance term as the invariance part of VICReg: the online tokens
        # (context encoder, unmasked window, same attention as the target encoder) of the window and of a copy with
        # iid noise N(0, inv_noise^2) added in the model input space (after window_relative) should agree:
        # mean over tokens of the squared difference of the L2-normalized tokens per dimension (scale of the JEPA
        # loss). Counteracts the noise amplification learned with the VC term (Vault modellumbau_rssm_mtsjepa,
        # 2026-11-13). Dropout handling as the VC pass (vc_no_dropout). Training only.
        if inv_weight < 0 or inv_noise <= 0:
            raise ValueError("inv_weight >= 0 and inv_noise > 0 required")
        if inv_weight > 0 and not self.group_tokens:
            raise ValueError("inv_weight is implemented for group_tokens only")
        self.inv_weight, self.inv_noise = float(inv_weight), float(inv_noise)
        # inv_noise_scale (opt-in): "none" = the same noise sd inv_noise in every channel (model units, i.e. 1 m
        # position jitter but below the step noise of gyros and radio); "step" = per channel inv_noise times the sd
        # of the step (raster step) within a window in the model input space, inv_noise_sd, computed once on training
        # windows in train.py and stored in the config (Vault modellumbau_rssm_mtsjepa, 2026-11-14). Constant
        # channels (step sd 0) get no noise.
        if inv_noise_scale not in ("none", "step"):
            raise ValueError(f"inv_noise_scale must be none or step, not {inv_noise_scale!r}")
        if (inv_noise_scale == "step") != (inv_noise_sd is not None):
            raise ValueError("inv_noise_sd is given exactly for inv_noise_scale step")
        if inv_noise_sd is not None:
            if len(inv_noise_sd) != num_features or min(inv_noise_sd) < 0:
                raise ValueError("inv_noise_sd: one sd >= 0 per feature required")
            self.register_buffer("inv_sd", torch.tensor(inv_noise_sd, dtype=torch.float32), persistent=False)
        else:
            self.inv_sd = None
        # target_no_dropout (opt-in): the EMA target branch always runs in eval mode (no dropout noise in the
        # targets), as the momentum targets of BYOL/DINO. Default: the target follows the model's train/eval mode.
        self.target_no_dropout = bool(target_no_dropout)
        if self.target_no_dropout:
            self.train(self.training)
        # recon_weight (opt-in, default 0): reconstruction head per group (option 1, Vault modellumbau_rssm_mtsjepa,
        # 2026-11-16). A linear map of the predictor output of every token (hidden and visible) back to the patch
        # values of its group in the model input space (after window_relative); weighted MSE added to the loss, the
        # gradient reaches predictor, context encoder and group embeddings (the target branch follows by EMA). Lets
        # the tokens keep the values (the probe upper bound of the plain JEPA was 13-42 m for the position) and gives
        # a value-space prediction for hidden groups (recon_values). Training only; the validation loss stays JEPA.
        if recon_weight < 0:
            raise ValueError("recon_weight >= 0 required")
        if recon_weight > 0 and not self.group_tokens:
            raise ValueError("recon_weight is implemented for group_tokens only")
        self.recon_weight = float(recon_weight)
        if self.recon_weight > 0:
            self.recon_head = nn.ModuleList(nn.Linear(embed_dim, patch_length * len(g)) for g in self.groups)
        # group_mask_max (opt-in, default 1): the group mask hides 1..n distinct groups (uniform in n) instead of one,
        # lever e (as rssm group_mask_max). 1 = previous behaviour (same random draws).
        if int(group_mask_max) < 1 or (self.group_tokens and int(group_mask_max) > self.n_groups):
            raise ValueError("group_mask_max must be in 1..n_groups")
        if int(group_mask_max) > 1 and not self.group_tokens:
            raise ValueError("group_mask_max is implemented for group_tokens only")
        self.group_mask_max = int(group_mask_max)
        # context_only_groups (opt-in, way 1, Vault modellumbau_rssm_mtsjepa 2026-11-17): groups that stay input
        # (context) but are never hidden and never a prediction target. Measured on normal flights: network_up/down and
        # SCALED_PRESSURE are not predictable from the other groups by any predictor (pred_ceiling.py, 2026-11-17);
        # their group score is a pure novelty measure. Empty = previous behaviour (same random draws).
        ctx = list(context_only_groups or [])
        if ctx and not self.group_tokens:
            raise ValueError("context_only_groups is implemented for group_tokens only")
        if ctx:
            if not self.group_names:
                raise ValueError("context_only_groups needs group_names")
            unknown = [g for g in ctx if g not in self.group_names]
            if unknown:
                raise ValueError(f"context_only_groups: unknown groups {unknown}")
            co = torch.tensor([g in ctx for g in self.group_names])
            if int((~co).sum()) < self.group_mask_max:
                raise ValueError("context_only_groups leaves fewer target groups than group_mask_max")
        else:
            co = torch.zeros(self.n_groups if self.group_tokens else 0, dtype=torch.bool)
        self.context_only = bool(ctx)
        self.register_buffer("ctx_only", co, persistent=False)
        # group_mask_sets (opt-in, default off): with group_mask_set_prob a masked sample hides one of these sets of
        # groups (uniform) instead of the usual random group(s); off = previous behaviour (same random draws).
        self.group_mask_set_prob = float(group_mask_set_prob)
        if not 0.0 <= self.group_mask_set_prob <= 1.0:
            raise ValueError("group_mask_set_prob must be in [0, 1]")
        sets = [list(s) for s in (group_mask_sets or [])]
        if self.group_mask_set_prob > 0:
            if not sets or any(not s for s in sets):
                raise ValueError("group_mask_set_prob > 0 needs non-empty group_mask_sets")
            if not self.group_names:
                raise ValueError("group_mask_sets needs group_names")
            unknown = sorted({g for s in sets for g in s if g not in self.group_names})
            if unknown:
                raise ValueError(f"group_mask_sets: unknown groups {unknown}")
            if not self.group_tokens:
                raise ValueError("group_mask_sets is implemented for group_tokens only")
            if set(ctx) & {g for s_ in sets for g in s_}:
                raise ValueError("group_mask_sets must not contain context_only_groups")
            sm = torch.tensor([[g in s for g in self.group_names] for s in sets], dtype=torch.bool)
        else:
            sm = torch.zeros(0, len(self.group_names or []), dtype=torch.bool)
        self.register_buffer("group_set_masks", sm, persistent=False)
        # group_mask_start_max (opt-in, default 0): the group mask (mask kind "group", also the joint mask) hides its
        # group(s) from a random patch p ~ U{0..group_mask_start_max} on instead of in all patches; trains the
        # transition visible -> hidden (Vault modellumbau_rssm_mtsjepa, joint mask analysis 2026-11-20)
        self.group_mask_start_max = int(group_mask_start_max)
        if not 0 <= self.group_mask_start_max < window_length // patch_length:
            raise ValueError("group_mask_start_max must be in [0, num_patches)")
        if self.group_mask_start_max > 0 and not self.group_tokens:
            raise ValueError("group_mask_start_max is implemented for group_tokens only")
        # consistency_sets (opt-in, scoring only): sets of groups scored by set_scores (the set hidden in all patches,
        # its tokens predicted from the other groups); no effect on training or anomaly_score
        self.consistency_sets = [list(c) for c in (consistency_sets or [])]
        # group_mask_set_mode (with group_mask_set_prob): "replace" (default, as before) = a group-kind window hides the
        # set instead of its single group, so every group is alone the target only half as often with prob 0.5 (gate
        # analysis 2026-11-20). "add" = the set is a fourth mask kind with the same share as before (group share x
        # group_mask_set_prob), taken proportionally from the temporal and causal kinds; the group kind keeps its share
        # and always hides single groups.
        if group_mask_set_mode not in ("replace", "add"):
            raise ValueError("group_mask_set_mode must be replace or add")
        self.group_mask_set_mode = group_mask_set_mode
        if group_mask_set_mode == "add" and self.group_mask_set_prob > 0:
            t_, g_, c_ = (float(v) for v in self.mask_mix)
            s_ = g_ * self.group_mask_set_prob
            if t_ + c_ <= s_:
                raise ValueError("group_mask_set_mode add: temporal + causal share too small for the set share")
            q_ = s_ / (t_ + c_)
            self.register_buffer("mask_mix4", torch.tensor([t_ * (1 - q_), g_, c_ * (1 - q_), s_]), persistent=False)
        unknown = sorted({g for c in self.consistency_sets for g in c if g not in (self.group_names or [])})
        if unknown:
            raise ValueError(f"consistency_sets: unknown groups {unknown}")
        # fourier_weight (opt-in, default 0, way 3, Vault modellumbau_rssm_mtsjepa 2026-11-17): auxiliary target in a
        # fixed, non-learned second view, after Zhu et al. 2026 (arXiv:2609.04264, random Fourier features of the
        # physical state, shallow head, weight 0.1) and the cross-prediction of JEPA-x (Wen et al. 2026,
        # arXiv:2608.24044: a target that encoder and predictor cannot co-adapt). Per group g and patch the target is
        # y = [cos(W_g^T v), sin(W_g^T v)] of the patch values v (model input space, P x F_g), W_g fixed random
        # (seed fourier_seed), columns in four frequency bands fourier_scale * {1, 2, 4, 8} / sqrt(P * F_g), so that
        # also small differences (e.g. GPS - EKF of a few metres) change the target. A shallow head per group
        # (D -> 64 -> 2 * fourier_dim) predicts y (MSE): on the predictor output of the hidden tokens (cross-
        # prediction of what is hidden) and, with fourier_encoder, on the context-encoder output of the visible tokens
        # (grounding of the online tokens, which the target encoder follows by EMA). Training only; the validation
        # loss stays the JEPA loss. Unlike recon_weight: bounded, non-linear fixed target, hidden tokens only on the
        # predictor side, plus the encoder side.
        if fourier_weight < 0 or int(fourier_dim) < 4 or int(fourier_dim) % 4 or fourier_scale <= 0:
            raise ValueError("fourier_weight >= 0, fourier_dim a positive multiple of 4, fourier_scale > 0 required")
        if fourier_weight > 0 and not self.group_tokens:
            raise ValueError("fourier_weight is implemented for group_tokens only")
        self._init_step1(embed_dim, predict_var, var_weight, self_head, self_weight, self_context, score_mode)
        self.fourier_weight, self.fourier_encoder = float(fourier_weight), bool(fourier_encoder)
        if self.fourier_weight > 0:
            gen = torch.Generator().manual_seed(int(fourier_seed)); d = int(fourier_dim)
            band = torch.tensor([1.0, 2.0, 4.0, 8.0]).repeat_interleave(d // 4)
            for gi, g in enumerate(self.groups):
                k = patch_length * len(g)
                w = torch.randn(k, d, generator=gen) * band.view(1, -1) * float(fourier_scale) / k ** 0.5
                self.register_buffer(f"fw_{gi}", w)
            self.fourier_head = nn.ModuleList(nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 2 * d))
                                              for _ in self.groups)
        self._init_ac(ac, ac_action_group, ac_radio_groups, ac_horizon, ac_group_mask_prob, embed_dim,
                      ac_use_action, ac_one_way_radio)
        self.ac_score = str(ac_score)
        if self.ac_score not in ("token", "recon", "recon_norm"):
            raise ValueError("ac_score must be token, recon or recon_norm")
        if self.ac_score != "token" and not (self.ac and self.recon_weight > 0):
            raise ValueError("ac_score recon / recon_norm needs ac and a reconstruction head (recon_weight > 0)")
        if self.ac_score == "recon_norm":
            if not ac_recon_norm or len(ac_recon_norm.get("med", [])) != num_features or len(ac_recon_norm.get("scale", [])) != num_features:
                raise ValueError("ac_score recon_norm needs ac_recon_norm with med and scale per feature")
            if min(ac_recon_norm["scale"]) <= 0:
                raise ValueError("ac_recon_norm scale must be > 0")
            self.register_buffer("ac_rn_med", torch.tensor(ac_recon_norm["med"], dtype=torch.float32), persistent=False)
            self.register_buffer("ac_rn_scale", torch.tensor(ac_recon_norm["scale"], dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------
    # MTS-JEPA-AC (stage 1, Vault Training/entwurf_mts_jepa_ac, approved Lasse 2026-11-26)
    # ------------------------------------------------------------------

    def _init_ac(self, ac, action_group, radio_groups, horizon, group_mask_prob, embed_dim, use_action=True,
                 one_way_radio=True):
        """ac (opt-in): action-conditioned, causal variant. Token order (patch n, group g) -> n * G + g.
        Context and target encoder: block-causal in time (a token sees patches <= its own), one-way radio (physical
        tokens never attend radio tokens; radio tokens may attend physical ones); the target encoder additionally only
        within its group (target_attn group). Predictor: one pass over [context tokens (N*G) | raw action tokens (N) |
        queries (H*N*G)]; the query of horizon k, patch m, group g predicts the target token (m, g) from the context of
        patches <= m - k and the raw action tokens of patches m - k + 1 .. m (the action group itself: .. m - 1, i.e. a
        forward prediction of the command), and sees the other queries of the same k and m (same radio rule). Direct
        multi-step prediction from the cut-off m - k (all horizons in parallel, no recursive feedback), so a lasting
        offset is measured against a prediction anchored before it. Training: all groups are targets; with probability
        ac_group_mask_prob a window hides one random group in all patches of the context (sensor redundancy, as the
        group mask of the masked JEPA; also used by the evaluation tools). Loss = normalized token error over the valid
        tokens (m >= k) of all horizons; recon_weight puts the reconstruction head (decoder anchor) on the horizon-1
        predictions; vc/inv terms as before with the causal target attention. The masked-JEPA interface
        (token_errors, predict_tokens, target_tokens, recon_values, set_scores) works with the horizon-1 predictions:
        mask = tokens hidden from the context."""
        self.ac = bool(ac); self.ac_use_action = bool(use_action); self.ac_one_way_radio = bool(one_way_radio)
        if not self.ac:
            return
        if not self.group_tokens or not self.group_names:
            raise ValueError("ac needs group_tokens with group_names")
        if self.target_attn != "group":
            raise ValueError("ac is implemented for target_attn group")
        if self.context_only or self.group_mask_set_prob > 0 or self.fourier_weight > 0 or self.var_head is not None \
                or self.self_head_on:
            raise ValueError("ac stage 1 does not combine with context_only_groups, group_mask_sets, fourier, step-1 heads")
        names = list(self.group_names); N, G = self.num_patches, self.n_groups
        if action_group not in names:
            raise ValueError(f"ac_action_group {action_group!r} not in {names}")
        radio_groups = ["radio_up", "radio_down"] if radio_groups is None else list(radio_groups)
        self.ac_a = names.index(action_group)
        self.ac_h = int(horizon)
        if not 1 <= self.ac_h < N:
            raise ValueError("ac_horizon must be in 1..num_patches-1")
        self.ac_gmp = float(group_mask_prob)
        if not 0.0 <= self.ac_gmp <= 1.0:
            raise ValueError("ac_group_mask_prob must be in [0, 1]")
        radio = torch.tensor([g in radio_groups for g in names])
        L = N * G; n_of = torch.arange(L) // G; g_of = torch.arange(L) % G
        phys_q = ~radio[g_of]; rad_k = radio[g_of]
        ow = (phys_q.view(-1, 1) & rad_k.view(1, -1)) if self.ac_one_way_radio else torch.zeros(L, L, dtype=torch.bool)
        ctx = (n_of.view(1, -1) > n_of.view(-1, 1)) | ow                                         # True = blocked
        self.register_buffer("ac_ctx_mask", ctx, persistent=False)
        self.register_buffer("ac_tgt_mask", ctx | (g_of.view(-1, 1) != g_of.view(1, -1)), persistent=False)
        H = self.ac_h; Q = H * N * G; T_ = L + N + Q
        m = torch.ones(T_, T_, dtype=torch.bool)
        m[:L, :L] = ctx                                            # context rows: as in the encoder
        m[L:L + N, L:L + N] = ~torch.eye(N, dtype=torch.bool)      # action rows: only themselves
        qk = torch.arange(Q) // (N * G); qn = (torch.arange(Q) // G) % N; qg = torch.arange(Q) % G
        qphys = ~radio[qg]
        cut = qn - (qk + 1)                                        # last context patch of the query
        m[L + N:, :L] = (n_of.view(1, -1) > cut.view(-1, 1)) | ((qphys.view(-1, 1) & rad_k.view(1, -1))
                                                                if self.ac_one_way_radio else False)
        an = torch.arange(N)
        last_act = torch.where(qg == self.ac_a, qn - 1, qn)        # the command itself is not given to its own query
        m[L + N:, L:L + N] = ~((an.view(1, -1) > cut.view(-1, 1)) & (an.view(1, -1) <= last_act.view(-1, 1)))
        if not self.ac_use_action:
            m[L + N:, L:L + N] = True                              # ablation: queries see no action tokens
        same = (qk.view(-1, 1) == qk.view(1, -1)) & (qn.view(-1, 1) == qn.view(1, -1))
        m[L + N:, L + N:] = ~same | ((qphys.view(-1, 1) & radio[qg].view(1, -1)) if self.ac_one_way_radio else False)
        assert not m.all(1).any(), "a token without any key"
        self.register_buffer("ac_pred_mask", m, persistent=False)
        valid = torch.arange(N).view(1, N, 1) >= torch.arange(1, H + 1).view(H, 1, 1)  # (H, N, 1): m >= k
        self.register_buffer("ac_valid", valid.expand(H, N, G).clone(), persistent=False)
        self.ac_query = nn.Parameter(torch.zeros(H, embed_dim)); nn.init.normal_(self.ac_query, std=0.02)
        self.ac_act_pos = nn.Parameter(torch.zeros(1, 1, embed_dim)); nn.init.normal_(self.ac_act_pos, std=0.02)

    def _tgt_attn(self):
        """Attention mask of the target encoder and of its online counterpart (vc/inv terms)."""
        if getattr(self, "ac", False):
            return self.ac_tgt_mask
        return self.target_attn_mask if self.target_attn == "group" else None

    def _ac_forward(self, batch: torch.Tensor, hide: torch.Tensor | None = None):
        """(predictions (B, H, N, G, D), context (B, N, G, D)); hide (B, N, G) True = token hidden from the context
        (for the action group also its raw action token)."""
        tok = self._group_tokens(batch, target=False)                              # (B, N, G, D)
        B, N, G, D = tok.shape; H = self.ac_h
        mtok = (self.mask_token.view(1, 1, 1, -1) + self.gpos_embed + self.group_pos).expand(B, -1, -1, -1)
        tin = tok if hide is None else torch.where(hide.unsqueeze(-1), mtok, tok)
        ctx = self.context_encoder(tin.reshape(B, N * G, D), mask=self.ac_ctx_mask)
        act = tin[:, :, self.ac_a] + self.ac_act_pos                               # (B, N, D)
        q = (self.mask_token.view(1, 1, 1, 1, -1) + self.gpos_embed.unsqueeze(1) + self.group_pos.unsqueeze(1)
             + self.ac_query.view(1, H, 1, 1, -1)).expand(B, -1, N, G, -1)
        seq = torch.cat([ctx, act, q.reshape(B, H * N * G, D)], dim=1)
        out = self.predictor(seq, mask=self.ac_pred_mask)
        return out[:, N * G + N:].reshape(B, H, N, G, D), ctx.view(B, N, G, D)

    def _ac_errors(self, batch: torch.Tensor, hide: torch.Tensor | None = None):
        """(err (B, H, N, G) normalized token error per horizon, preds, ctx)."""
        preds, ctx = self._ac_forward(batch, hide)
        t = F.normalize(self.target_tokens(batch), dim=-1).unsqueeze(1)
        err = ((F.normalize(preds, dim=-1) - t) ** 2).mean(dim=-1)
        return err, preds, ctx

    def _ac_training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, G = batch.shape[0], self.num_patches, self.n_groups
        hide = None
        if self.training and self.ac_gmp > 0:
            on = torch.rand(B, device=batch.device) < self.ac_gmp
            g = torch.randint(0, G, (B,), device=batch.device)
            hide = (torch.arange(G, device=batch.device).view(1, 1, G) == g.view(B, 1, 1)) & on.view(B, 1, 1)
            hide = hide.expand(B, N, G)
        err, preds, _ = self._ac_errors(batch, hide)
        v = self.ac_valid.unsqueeze(0).expand(B, -1, -1, -1)
        loss = err[v].mean()
        out = {"loss": loss, "jepa": loss.detach()}
        for k in range(self.ac_h):
            out[f"jepa_h{k + 1}"] = err[:, k][v[:, k]].mean().detach()
        if self.recon_weight > 0 and self.training:
            se = (self.recon_from_pred(preds[:, 0]) - self.model_input(batch)).pow(2)[:, self.patch_length:]
            rec = se.mean(); out["loss"] = out["loss"] + self.recon_weight * rec; out["recon"] = rec.detach()
        if self.vc_weight > 0 and self.training:
            var, cov = self.vc_terms(batch)
            out["loss"] = out["loss"] + self.vc_weight * (var + self.vc_cov_ratio * cov)
            out["vc_var"] = var.detach(); out["vc_cov"] = cov.detach()
        if self.inv_weight > 0 and self.training:
            inv = self.inv_term(batch); out["loss"] = out["loss"] + self.inv_weight * inv; out["inv"] = inv.detach()
        return out

    @torch.no_grad()
    def _ac_recon_scores(self, batch: torch.Tensor, mask: torch.Tensor | None):
        """ac_score recon: (score, valid), both (B, N, G); score = squared error of the reconstruction head on the
        horizon-1 prediction (mask hidden from the context) vs the model input, mean over the patch's time steps and
        the group's channels; valid = patches >= 1 (and, with a mask, the hidden tokens)."""
        pred = self._ac_forward(batch, mask)[0][:, 0]
        se = (self.recon_from_pred(pred) - self.model_input(batch)).pow(2)
        B, N, G, P = batch.shape[0], self.num_patches, self.n_groups, self.patch_length
        se = se.reshape(B, N, P, -1).mean(2)                                         # (B, N, F)
        if self.ac_score == "recon_norm":
            se = (se - self.ac_rn_med) / self.ac_rn_scale
        sc = torch.stack([se[:, :, getattr(self, f"gidx_{g}")].mean(-1) for g in range(G)], dim=-1)
        v = (torch.arange(N, device=batch.device) >= 1).view(1, N, 1).expand(B, N, G)
        if mask is not None:
            v = v & mask
        return sc, v

    @torch.no_grad()
    def _ac_score(self, batch: torch.Tensor) -> torch.Tensor:
        """Window score: mean normalized token error over all groups, horizons and valid patches (nothing hidden);
        ac_score recon: mean reconstruction error over all groups and patches >= 1 (nothing hidden)."""
        if getattr(self, "ac_score", "token") in ("recon", "recon_norm"):
            sc, v = self._ac_recon_scores(batch, None); v = v.float()
            return (sc * v).sum((1, 2)) / v.sum((1, 2))
        err = self._ac_errors(batch)[0]
        v = self.ac_valid.unsqueeze(0).float()
        return (err * v).sum((1, 2, 3)) / v.sum((1, 2, 3))

    @torch.no_grad()
    def _ac_val_stats(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """val_stats with the horizon-1 errors of all groups at patches >= 1, nothing hidden."""
        B = batch.shape[0]
        err = self._ac_errors(batch)[0][:, 0]                                       # (B, N, G)
        v = self.ac_valid[0].unsqueeze(0).expand(B, -1, -1).float()
        tgt = F.normalize(self.target_tokens(batch), dim=-1)
        return {"err_sum": (err * v).sum((0, 1)), "err_n": v.sum((0, 1)),
                "t_sum": tgt.sum((0, 1)), "t_n": torch.full((self.n_groups,), float(B * self.num_patches), device=batch.device)}

    # ------------------------------------------------------------------
    # EMA handling
    # ------------------------------------------------------------------

    def _init_step1(self, embed_dim, predict_var, var_weight, self_head, self_weight, self_context, score_mode):
        """Step 1 of Vault modellumbau_rssm_mtsjepa (2026-11-18), uncertainty-aware score, opt-in (group_tokens only).
        predict_var: per group a head (D -> 64 -> 1) on the DETACHED predictor output -> log-variance of the token error,
          trained with the NLL 0.5 * (log var + err / var) on the DETACHED error of the hidden tokens (var_weight): the
          JEPA itself learns exactly as without it.
        self_head: for every context-only group (way 1) a head that predicts the group's target token at patch n from
          its own target tokens at patches n-self_context..n-1 (temporal own-history prediction, D*k -> 2D -> D), on
          detached target tokens (self_weight), error as the token error (normalized MSE); with predict_var also a
          variance head for it. So groups that cannot be predicted from the other groups still enter the score.
        score_mode "nll": anomaly_score = mean NLL over the scored tokens (eval mask of the target groups + self-head
          tokens of the context groups at the eval-mask patches); "mse" (default) unchanged."""
        if score_mode not in ("mse", "nll"):
            raise ValueError(f"score_mode must be 'mse' or 'nll', got {score_mode!r}")
        if (predict_var or self_head or score_mode == "nll") and not self.group_tokens:
            raise ValueError("predict_var, self_head and score_mode 'nll' are implemented for group_tokens only")
        if score_mode == "nll" and not predict_var:
            raise ValueError("score_mode 'nll' needs predict_var")
        if self_head and not self.context_only:
            raise ValueError("self_head needs context_only_groups")
        if not 1 <= int(self_context) < self.num_patches:
            raise ValueError("self_context in [1, num_patches) required")
        self.predict_var, self.var_weight, self.score_mode = bool(predict_var), float(var_weight), score_mode
        self.self_head_on, self.self_weight, self.self_context = bool(self_head), float(self_weight), int(self_context)
        G = self.n_groups if self.group_tokens else 0
        mk = lambda: nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.var_head = nn.ModuleList(mk() for _ in range(G)) if self.predict_var else None
        if self.self_head_on:
            k = self.self_context
            self.ctx_groups = [g for g in range(G) if bool(self.ctx_only[g])]
            self.self_net = nn.ModuleDict({str(g): nn.Sequential(nn.Linear(k * embed_dim, 2 * embed_dim), nn.GELU(),
                                                                 nn.Linear(2 * embed_dim, embed_dim)) for g in self.ctx_groups})
            self.self_var = (nn.ModuleDict({str(g): mk() for g in self.ctx_groups}) if self.predict_var else None)

    def _self_pred(self, tgt: torch.Tensor):
        """Own-history prediction of the context groups: per context group (pred, lv or None) on patches k..N-1."""
        k, out = self.self_context, {}
        for g in self.ctx_groups:
            t = tgt[:, :, g].detach()                                                   # (B, N, D)
            hist = torch.cat([t[:, i:t.shape[1] - k + i] for i in range(k)], dim=-1)   # (B, N-k, k*D)
            p = self.self_net[str(g)](hist)
            lv = self.self_var[str(g)](p.detach()).squeeze(-1).clamp(-10.0, 8.0) if self.self_var is not None else None
            out[g] = (p, lv)
        return out

    def step1_terms(self, batch: torch.Tensor, mask: torch.Tensor, err: torch.Tensor, pred: torch.Tensor):
        """Per-token score terms (B, N, G) and validity (B, N, G): NLL (score_mode nll) or the error (mse), for the
        hidden target tokens and, with self_head, for the context groups from patch self_context on; plus the
        training losses of the step-1 heads."""
        B, N, G = mask.shape; score = torch.zeros_like(err); valid = mask.clone(); losses = {}
        if self.var_head is not None:
            lv = torch.stack([self.var_head[g](pred[:, :, g].detach().float()).squeeze(-1) for g in range(G)], -1).clamp(-10.0, 8.0)
            nll = 0.5 * (lv + err.detach() * torch.exp(-lv))
            losses["var_nll"] = nll[mask].mean() if mask.any() else nll.sum() * 0
            score = 0.5 * (lv + err * torch.exp(-lv)) if self.score_mode == "nll" else err
        else:
            score = err
        if self.self_head_on:
            tgt = F.normalize(self.target_tokens(batch), dim=-1); k = self.self_context; se_all = []
            for g, (p, lv) in self._self_pred(tgt).items():
                e = ((F.normalize(p, dim=-1) - tgt[:, k:, g]) ** 2).mean(-1)            # (B, N-k)
                se_all.append(e)
                s = 0.5 * (lv + e * torch.exp(-lv)) if (self.score_mode == "nll" and lv is not None) else e
                if lv is not None:
                    losses.setdefault("self_nll", []).append((0.5 * (lv + e.detach() * torch.exp(-lv))).mean())
                score = score.clone(); score[:, k:, g] = s; valid[:, :k, g] = False
            losses["self_err"] = torch.stack([x.mean() for x in se_all]).mean()
            if "self_nll" in losses: losses["self_nll"] = torch.stack(losses["self_nll"]).mean()
        return score, valid, losses

    def _target_modules(self):
        mods = [
            (self.patch_embed, self.target_patch_embed),
            (self.context_encoder, self.target_encoder),
        ]
        if getattr(self, "group_tokens", False):
            mods.append((self.group_embed, self.target_group_embed))
        return mods

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

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "target_no_dropout", False):
            for _, target in self._target_modules():
                target.eval()
        return self

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

    # --- group tokens -----------------------------------------------

    def _group_tokens(self, x: torch.Tensor, target: bool, noise_sd: float | torch.Tensor = 0.0) -> torch.Tensor:
        """(B, T, F) -> tokens (B, N, G, D) incl. time and group position. noise_sd > 0 (scalar or per feature (F,)):
        iid noise added in the model input space (after window_relative), for the invariance term."""
        if self.win_rel is not None:
            x = self.win_rel(x)
        if torch.is_tensor(noise_sd) or noise_sd > 0:
            x = x + noise_sd * torch.randn_like(x)
        B = x.shape[0]; P = self.patch_length; N = self.num_patches
        emb = self.target_group_embed if target else self.group_embed
        toks = []
        for gi in range(self.n_groups):
            xg = x.index_select(2, getattr(self, f"gidx_{gi}"))           # (B, T, Fg)
            toks.append(emb[gi](xg.reshape(B, N, P * xg.shape[2])))        # (B, N, D)
        return torch.stack(toks, dim=2) + self.gpos_embed + self.group_pos

    def sample_token_mask(self, B: int, device) -> torch.Tensor:
        """Training mask (B, N, G), True = masked, one mask type per window from mask_mix."""
        N, G = self.num_patches, self.n_groups
        add = self.group_mask_set_prob > 0 and self.group_mask_set_mode == "add"
        kind = torch.multinomial(self.mask_mix4 if add else self.mask_mix, B, replacement=True)
        m = torch.zeros(B, N, G, dtype=torch.bool, device=device)
        tmask = self._random_mask(B, device)                                # temporal
        m |= (kind == 0).view(B, 1, 1) & tmask.unsqueeze(-1)
        if self.group_mask_max == 1 and not self.context_only:
            g = torch.randint(G, (B,), device=device)                       # group
            gm = torch.zeros(B, G, dtype=torch.bool, device=device); gm[torch.arange(B, device=device), g] = True
        elif self.group_mask_max == 1:                                      # group among the target groups
            allowed = torch.nonzero(~self.ctx_only).squeeze(1)
            g = allowed[torch.randint(len(allowed), (B,), device=device)]
            gm = torch.zeros(B, G, dtype=torch.bool, device=device); gm[torch.arange(B, device=device), g] = True
        else:                                                               # 1..n distinct groups
            n = torch.randint(1, self.group_mask_max + 1, (B, 1), device=device)
            r = torch.rand(B, G, device=device)
            if self.context_only:
                r = r.masked_fill(self.ctx_only.view(1, -1), 2.0)          # context-only groups rank last
            rank = r.argsort(1).argsort(1)
            gm = rank < n
        if add:                                                            # joint mask as fourth kind (kind 3)
            j = torch.randint(self.group_set_masks.shape[0], (B,), device=device)
            sm = (kind == 3).view(B, 1, 1) & self.group_set_masks[j].unsqueeze(1)
            if self.group_mask_start_max > 0:
                p3 = torch.randint(0, self.group_mask_start_max + 1, (B,), device=device)
                sm = sm & (torch.arange(N, device=device).unsqueeze(0) >= p3.unsqueeze(1)).unsqueeze(-1)
            m |= sm
        elif self.group_mask_set_prob > 0:                                 # joint mask: one of group_mask_sets
            use = torch.rand(B, device=device) < self.group_mask_set_prob
            j = torch.randint(self.group_set_masks.shape[0], (B,), device=device)
            gm = torch.where(use.unsqueeze(1), self.group_set_masks[j], gm)
        if self.group_mask_start_max > 0:  # hidden from a random patch on
            p0 = torch.randint(0, self.group_mask_start_max + 1, (B,), device=device)
            pm = torch.arange(N, device=device).unsqueeze(0) >= p0.unsqueeze(1)  # (B, N)
            m |= (kind == 1).view(B, 1, 1) & gm.unsqueeze(1) & pm.unsqueeze(-1)
        else:
            m |= (kind == 1).view(B, 1, 1) & gm.unsqueeze(1)
        k = torch.randint(1, self.causal_max + 1, (B,), device=device)      # causal: last k patches
        cm = torch.arange(N, device=device).unsqueeze(0) >= (N - k).unsqueeze(1)
        m |= (kind == 2).view(B, 1, 1) & cm.unsqueeze(-1)
        if self.context_only:
            m &= ~self.ctx_only.view(1, 1, -1)
        return m

    def score_token_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Scored tokens: the hidden target tokens; with self_head also the context groups at the hidden patches.
        Used by anomaly_score (step 1); valid only from patch self_context on for the context groups."""
        if not self.self_head_on:
            return mask
        patches = mask.any(-1, keepdim=True)
        return mask | (patches & self.ctx_only.view(1, 1, -1))

    @torch.no_grad()
    def token_scores(self, batch: torch.Tensor, mask: torch.Tensor):
        """(score, valid), both (B, N, G): the per-token terms of anomaly_score with `mask` hidden from the context
        (step 1 aware; with self_head the context groups are scored at the hidden patches, as in anomaly_score).
        ac with ac_score recon: reconstruction error of the hidden tokens (_ac_recon_scores)."""
        if getattr(self, "ac", False) and getattr(self, "ac_score", "token") in ("recon", "recon_norm"):
            return self._ac_recon_scores(batch, mask)
        err, pred = self._token_errors_pred(batch, mask)
        if self.score_mode == "nll" or self.self_head_on:
            s, v, _ = self.step1_terms(batch, self.score_token_mask(mask), err, pred)
            return s, v
        return err, mask

    def eval_token_mask(self, B: int, device) -> torch.Tensor:
        m = self.eval_mask.view(1, -1, 1).expand(B, -1, self.n_groups).to(device)
        return m & ~self.ctx_only.view(1, 1, -1) if self.context_only else m

    @torch.no_grad()
    def target_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """Target-encoder output per token, (B, N, G, D), not normalized."""
        B, N, G = batch.shape[0], self.num_patches, self.n_groups
        mask = self._tgt_attn()
        return self.target_encoder(self._group_tokens(batch, target=True).reshape(B, N * G, -1), mask=mask).view(B, N, G, -1)

    def vc_terms(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance and covariance terms (see vc_weight) on the online tokens of the unmasked window."""
        B, N, G = batch.shape[0], self.num_patches, self.n_groups
        mask = self._tgt_attn()
        was = self.context_encoder.training
        if self.vc_no_dropout:
            self.context_encoder.eval()
        try:
            z = self.context_encoder(self._group_tokens(batch, target=False).reshape(B, N * G, -1), mask=mask)
        finally:
            self.context_encoder.train(was)
        z = z.view(B, N, G, -1)
        if self.vc_space == "normalized":
            z = F.normalize(z, dim=-1) * z.shape[-1] ** 0.5
        return self._vc(z)

    def inv_term(self, batch: torch.Tensor) -> torch.Tensor:
        """Noise invariance term (see inv_weight) on the online tokens of the unmasked window."""
        B, N, G = batch.shape[0], self.num_patches, self.n_groups
        mask = self._tgt_attn()
        was = self.context_encoder.training
        if self.vc_no_dropout:
            self.context_encoder.eval()
        try:
            z1 = self.context_encoder(self._group_tokens(batch, target=False).reshape(B, N * G, -1), mask=mask)
            sd = self.inv_noise * self.inv_sd if self.inv_sd is not None else self.inv_noise
            z2 = self.context_encoder(self._group_tokens(batch, target=False, noise_sd=sd).reshape(B, N * G, -1), mask=mask)
        finally:
            self.context_encoder.train(was)
        return (F.normalize(z1, dim=-1) - F.normalize(z2, dim=-1)).pow(2).mean()

    def _vc(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, N, G, D): statistics over the batch per (patch, group), averaged."""
        B, N, G, D = z.shape
        zc = z - z.mean(0, keepdim=True)
        std = torch.sqrt(zc.pow(2).sum(0) / max(B - 1, 1) + 1e-4)                    # (N, G, D)
        var = F.relu(self.vc_gamma - std).mean()
        zz = zc.permute(1, 2, 0, 3).reshape(N * G, B, D)
        cov = zz.transpose(1, 2) @ zz / max(B - 1, 1)                                 # (N*G, D, D)
        off = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
        return var, (off.pow(2).sum((1, 2)) / D).mean()

    @torch.no_grad()
    def val_stats(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Validation statistics per group (eval mask): sum and count of the token errors of the masked tokens,
        and sum and count of the L2-normalized target tokens (for the spread of the targets, see train.py
        select_metric norm_loss)."""
        if getattr(self, "ac", False):
            return self._ac_val_stats(batch)
        B = batch.shape[0]
        mask = self.eval_token_mask(B, batch.device)
        err = self.token_errors(batch, mask)
        tgt = F.normalize(self.target_tokens(batch), dim=-1)  # (B, N, G, D)
        return {"err_sum": (err * mask).sum((0, 1)), "err_n": mask.sum((0, 1)).float(),
                "t_sum": tgt.sum((0, 1)), "t_n": torch.full((self.n_groups,), float(B * self.num_patches), device=batch.device)}

    def _predict_ctx(self, batch: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(predictor output, context-encoder output), both (B, N, G, D); mask (B, N, G) True = hidden.
        ac: the horizon-1 predictions (patch 0 has no context, only its action token)."""
        if getattr(self, "ac", False):
            preds, ctx = self._ac_forward(batch, mask)
            return preds[:, 0], ctx
        B, N, G = mask.shape
        tok = self._group_tokens(batch, target=False)
        mtok = (self.mask_token.view(1, 1, 1, -1) + self.gpos_embed + self.group_pos).expand(B, -1, -1, -1)
        ctx_in = torch.where(mask.unsqueeze(-1), mtok, tok).reshape(B, N * G, -1)
        ctx = self.context_encoder(ctx_in)
        return self.predictor(ctx).view(B, N, G, -1), ctx.view(B, N, G, -1)

    def predict_tokens(self, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predictor output per token (B, N, G, D), not normalized; mask (B, N, G) True = hidden."""
        return self._predict_ctx(batch, mask)[0]

    def _token_errors_pred(self, batch: torch.Tensor, mask: torch.Tensor, with_ctx: bool = False):
        pred, ctx = self._predict_ctx(batch, mask)
        tgt = self.target_tokens(batch)
        p, t = F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1)
        err = ((p - t) ** 2).mean(dim=-1)
        return (err, pred, ctx) if with_ctx else (err, pred)

    @torch.no_grad()
    def fourier_targets(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Fixed second view (fourier_weight > 0): per group (B, N, 2 * fourier_dim) from the patch values in the model
        input space."""
        x = self.model_input(batch).float(); B = x.shape[0]; N, P = self.num_patches, self.patch_length
        out = []
        for gi in range(self.n_groups):
            v = x.index_select(2, getattr(self, f"gidx_{gi}")).reshape(B, N, -1)
            z = v @ getattr(self, f"fw_{gi}")
            out.append(torch.cat([torch.cos(z), torch.sin(z)], -1))
        return out

    def fourier_terms(self, batch: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, ctx: torch.Tensor):
        """(predictor side on hidden tokens, encoder side on visible tokens), MSE to the fixed Fourier targets."""
        ys = self.fourier_targets(batch)
        sp, np_, se, ne = pred.new_zeros(()), 0, pred.new_zeros(()), 0
        for gi, y in enumerate(ys):
            hid = mask[:, :, gi]
            if hid.any():
                d = (self.fourier_head[gi](pred[:, :, gi][hid]).float() - y[hid]).pow(2).mean(-1)
                sp, np_ = sp + d.sum(), np_ + d.numel()
            vis = ~hid
            if self.fourier_encoder and vis.any():
                d = (self.fourier_head[gi](ctx[:, :, gi][vis]).float() - y[vis]).pow(2).mean(-1)
                se, ne = se + d.sum(), ne + d.numel()
        return sp / max(np_, 1), se / max(ne, 1)

    @torch.no_grad()
    def set_scores(self, batch: torch.Tensor, sets: list[list[str]] | None = None) -> torch.Tensor:
        """Masked consistency score (B, S): per set (default consistency_sets) the set hidden in all patches, mean of
        the token scores (token_scores: error or NLL as anomaly_score) over the set's target tokens. Context-only groups
        of a set are hidden but not scored. Opt-in, scoring only (Vault modellumbau_rssm_mtsjepa, 2026-11-20)."""
        self.eval()
        sets = self.consistency_sets if sets is None else [list(c) for c in sets]
        if not sets or not self.group_tokens:
            raise ValueError("set_scores needs sets of group names and group_tokens")
        B, N, G = batch.shape[0], self.num_patches, self.n_groups; cols = []
        for c in sets:
            mk = torch.zeros(B, N, G, dtype=torch.bool, device=batch.device)
            for g in c:
                mk[:, :, list(self.group_names).index(g)] = True
            sc, v = self.token_scores(batch, mk)
            tm = mk & v.bool()
            if self.context_only:
                tm = tm & ~self.ctx_only.view(1, 1, -1)
            cols.append((sc * tm).sum((1, 2)) / tm.sum((1, 2)).clamp(min=1))
        return torch.stack(cols, dim=1)

    def token_errors(self, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per token: squared error of the L2-normalized prediction vs. target, (B, N, G);
        mask (B, N, G) True = hidden from the context."""
        return self._token_errors_pred(batch, mask)[0]

    def recon_from_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Reconstruction head (recon_weight > 0): predictor tokens (B, N, G, D) -> values (B, T, F) in the model input
        space (after window_relative)."""
        B, N, G, _ = pred.shape; P = self.patch_length
        out = pred.new_zeros(B, N, P, self.num_features)
        for gi in range(G):
            idx = getattr(self, f"gidx_{gi}")
            out[:, :, :, idx] = self.recon_head[gi](pred[:, :, gi]).view(B, N, P, len(idx)).to(out.dtype)
        return out.reshape(B, N * P, self.num_features)

    def model_input(self, batch: torch.Tensor) -> torch.Tensor:
        """Values in the model input space (window_relative applied), the target of the reconstruction head."""
        return self.win_rel(batch) if self.win_rel is not None else batch

    @torch.no_grad()
    def recon_values(self, batch: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(reconstruction, model input), both (B, T, F); mask (B, N, G) True = hidden."""
        return self.recon_from_pred(self.predict_tokens(batch, mask)), self.model_input(batch)

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if getattr(self, "ac", False):
            return self._ac_training_step(batch)
        if self.group_tokens:
            B = batch.shape[0]
            mask = self.sample_token_mask(B, batch.device) if self.training else self.eval_token_mask(B, batch.device)
            if self.fourier_weight > 0 and self.training:
                err, pred, ctx = self._token_errors_pred(batch, mask, with_ctx=True)
            else:
                err, pred = self._token_errors_pred(batch, mask)
            # same as F.mse_loss over the masked tokens' normalized embeddings
            loss = err[mask].mean()
            out = {"loss": loss}
            if self.recon_weight > 0 and self.training:
                se = (self.recon_from_pred(pred) - self.model_input(batch)).pow(2)          # (B, T, F)
                hid = mask.unsqueeze(2).expand(-1, -1, self.patch_length, -1).reshape(B, -1, self.n_groups)
                hid_f = pred.new_zeros(B, se.shape[1], self.num_features, dtype=torch.bool)
                for gi in range(self.n_groups):
                    hid_f[:, :, getattr(self, f"gidx_{gi}")] = hid[:, :, gi:gi + 1]
                rec = se.mean()
                out = {"loss": loss + self.recon_weight * rec, "jepa": loss.detach(), "recon": rec.detach(),
                       "recon_hid": se[hid_f].mean().detach() if hid_f.any() else rec.detach() * 0}
            if self.vc_weight > 0 and self.training:
                var, cov = self.vc_terms(batch)
                out = {**out, "loss": out["loss"] + self.vc_weight * (var + self.vc_cov_ratio * cov),
                       "jepa": loss.detach(), "vc_var": var.detach(), "vc_cov": cov.detach()}
            if self.inv_weight > 0 and self.training:
                inv = self.inv_term(batch)
                out["loss"] = out["loss"] + self.inv_weight * inv
                out["jepa"] = loss.detach(); out["inv"] = inv.detach()
            if self.fourier_weight > 0 and self.training:
                fp, fe = self.fourier_terms(batch, mask, pred, ctx)
                out["loss"] = out["loss"] + self.fourier_weight * (fp + fe)
                out["jepa"] = loss.detach(); out["four_pred"] = fp.detach(); out["four_enc"] = fe.detach()
            if self.var_head is not None or self.self_head_on:  # step-1 heads, all on detached inputs
                _, _, ls = self.step1_terms(batch, mask, err, pred)
                add = self.var_weight * ls.get("var_nll", 0) + self.self_weight * (ls.get("self_err", 0) + ls.get("self_nll", 0))
                out["loss"] = out["loss"] + add; out["jepa"] = loss.detach()
                for k_, v_ in ls.items(): out[k_] = v_.detach()
            return out
        # Random mask while training; the fixed eval mask in eval mode, so
        # validation loss is not subject to mask-sampling noise and is
        # comparable between epochs and runs.
        if self.training:
            mask = self._random_mask(batch.shape[0], batch.device)
        else:
            mask = self.eval_mask.unsqueeze(0).expand(batch.shape[0], -1)
        pred, target = self._forward_jepa(batch, mask)
        # L2-normalize before the loss (as in SimSiam/BYOL/DINO): raw MSE in
        # embedding space is not scale-invariant, and nothing else bounds
        # the embedding scale (patch_embed/LayerNorm gains/mask_token can
        # grow unchecked), so the loss keeps climbing even while online and
        # target track each other closely — the absolute error grows with
        # the shared scale, not with a real drop in prediction quality.
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = F.mse_loss(pred, target)
        return {"loss": loss}

    @torch.no_grad()
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        self.eval()
        B = batch.shape[0]
        if getattr(self, "ac", False):
            return self._ac_score(batch)
        if self.group_tokens:
            mask = self.eval_token_mask(B, batch.device)
            if self.score_mode == "nll" or self.self_head_on:
                err, pred = self._token_errors_pred(batch, mask)
                score, valid, _ = self.step1_terms(batch, self.score_token_mask(mask), err, pred)
                return (score * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2))
            err = self.token_errors(batch, mask)
            return (err * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
        mask = self.eval_mask.unsqueeze(0).expand(B, -1)
        patches = self._patchify(batch)
        tokens = self.patch_embed(patches) + self.pos_embed
        mask_tok = self.mask_token + self.pos_embed
        ctx_in = torch.where(mask.unsqueeze(-1), mask_tok.expand(B, -1, -1), tokens)
        pred = self.predictor(self.context_encoder(ctx_in))
        tgt = self.target_encoder(self.target_patch_embed(patches) + self.pos_embed)
        # Same normalization as training_step, so scores are on the same
        # (scale-independent) footing the model was actually trained on.
        pred = F.normalize(pred, dim=-1)
        tgt = F.normalize(tgt, dim=-1)
        err = ((pred - tgt) ** 2).mean(dim=-1)  # (B, N)
        # Mean prediction error over masked positions only
        return (err * mask).sum(dim=1) / mask.sum(dim=1)