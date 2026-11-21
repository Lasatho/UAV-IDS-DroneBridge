"""
Recurrent State-Space Model (RSSM) for time-series anomaly detection.

DreamerV3-style world model without actions (observation-only variant,
following Domberg & Schildbach 2025): the latent state combines a
deterministic GRU path h_t and a stochastic latent z_t.

    h_t = GRU(h_{t-1}, [z_{t-1}])                  (deterministic path)
    prior:     p(z_t | h_t)                        (transition prediction)
    posterior: q(z_t | h_t, x_t)                   (observation update)
    decoder:   x̂_t = D(h_t, z_t)                  (reconstruction)

Training loss = reconstruction MSE + KL balancing as in DreamerV3:
    L_dyn = max(free_nats, KL(sg(q) ‖ p))   — trains the prior (dynamics)
    L_rep = max(free_nats, KL(q ‖ sg(p)))   — trains the posterior
    L = recon + β_dyn · L_dyn + β_rep · L_rep
The stop-gradient separation guarantees the prior receives a direct
training signal (matching the posterior) instead of depending on a
shared KL term that free-nats clamping can silence entirely.

Anomaly score = per-window mean of the one-step prediction error using
the *prior* latent (the model predicts x_t before seeing it; deviation
from the learned dynamics is the anomaly signal). The first
`score_burn_in` steps are excluded: each window starts from h = z = 0, so
the early "predictions" only reflect the data variance (t = 0 error ~60x
the steady-state error), not the learned dynamics.

predict_increment (opt-in, default off): the decoder predicts the change
from the previous observation, x̂_t = x_{t-1} + D(h_t, z_t), for prior and
posterior (x_{-1} := x_0). Without it the decoder reproduces the absolute
level of slowly varying channels (position, GPS, pressure) far less
precisely than their change per step: on normal v2 even the posterior
reconstruction was ~80-135x worse than predicting the last value (Vault
bewertung_ids_explorativ, 2026-11-09).

group_mask_prob (opt-in, default 0): sensor-group masking. With this
probability a training sample (window or chunk) hides one channel group
(MAVLink message / radio direction, `groups` = channel indices per group)
from the encoder for the whole sample; the decoder still reconstructs
every channel. The model must infer the hidden group from the others,
i.e. learn the redundancy between sensors (e.g. GPS from EKF, IMU,
velocity). The encoder input gets one flag per group (1 = hidden), so a
hidden group is not mistaken for a normal value of 0. At evaluation a
group can be hidden on purpose (`gmask`): its prediction error then
measures whether it fits the other sensors (Vault
bewertung_ids_explorativ / modellumbau_rssm_mtsjepa, decided 2026-11-10).

hidden_base (opt-in, default "observed"): with predict_increment the
prediction of a hidden group used its own true previous value as base, so
masking only tested the change of the hidden signal, never its level
(a slow offset or drift stayed invisible). "own": the base of the hidden
channels is the model's own previous posterior prediction, starting from 0
(the training mean), so the hidden group contributes no information of its
own. Windows only (no carried state). The decoder does not see that base,
so it has to carry the sum of its own outputs in h; errors accumulate
(hidden GPS ~80x worse than "GPS = EKF", Vault modellumbau_rssm_mtsjepa,
2026-11-12). Two fixes:
"own_input": as "own", but the decoder also gets the base of the hidden
channels (0 for visible ones) and one flag per group (1 = hidden), so the
correction to the base is learnable. Windows only.
"level": hidden channels are predicted as absolute level by a separate
head L(h_t, z_t), without base and increment; visible channels unchanged.
Works with carried state as well.

Levers against the drift of the multi-step prior (opt-in, defaults = previous behaviour; Vault
modellumbau_rssm_mtsjepa, "Vorgehensweise: Evaluation der Hebel", 2026-11-15). The open-loop prediction is no
training target: the prior is trained only by the KL term, which free_nats silences in > 99 % of the steps.
free_nats_mode "dim" (a2): free bits per latent dimension (threshold free_nats / stochastic_dim each) instead of
one threshold on the KL summed over the dimensions, so dimensions above their share keep a gradient.
group_mask_max (e): with group masking, hide 1..group_mask_max distinct groups per masked sample (uniform), so
combinations (e.g. both position groups) are trained, not only one group at a time.
group_mask_sets / group_mask_set_prob (opt-in, default off): joint group mask. With group_mask_set_prob, a sample
chosen for group masking hides one of the listed sets of groups (uniform over the sets) instead of the usual random
group(s). Intended for groups that measure the same quantity (e.g. GLOBAL_POSITION_INT and GPS_RAW_INT): with one of
them visible, the model can copy it and never has to infer the hidden group from the rest of the vehicle (Vault
modellumbau_rssm_mtsjepa, physics checks 2026-11-19). Off = previous behaviour (same random draws).
overshoot_horizon D > 0 with overshoot_kl_weight (b1, latent overshooting as in PlaNet) and/or
overshoot_obs_weight (b2, observation overshooting): from overshoot_starts random start steps s of the window,
roll the prior D steps from the filter state (h_s, z_s) without observations; b1 adds
KL(sg(q_{s+d}) || p_{s+d|s}) (mean over d, floored at overshoot_free_nats, default 0), b2 adds the weighted MSE of
the accumulated prior prediction (x_s + sum of the decoded increments) against x_{s+d}. Needs predict_increment
for b2. Windows only.

Gaussian latents are used instead of DreamerV3's categorical latents:
they are ONNX-friendly and sufficient for low-dimensional telemetry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import AnomalyDetectionModel
from ..window_relative import WindowRelative


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(AnomalyDetectionModel):
    def __init__(
        self,
        num_features: int,
        window_length: int,
        stochastic_dim: int = 32,
        deterministic_dim: int = 128,
        hidden_dim: int = 256,
        kl_dyn_beta: float = 0.5,
        kl_rep_beta: float = 0.1,
        free_nats: float = 1.0,
        score_burn_in: int = 4,
        predict_increment: bool = False,
        group_mask_prob: float = 0.0,
        groups: list[list[int]] | None = None,
        group_names: list[str] | None = None,
        hidden_base: str = "observed",
        window_relative: dict | None = None,
        loss_weight: str = "none",
        loss_weights: list[float] | None = None,
        loss_weight_floor: float = 0.0,
        abs_inputs: list[str] | None = None,
        abs_input_idx: list[int] | None = None,
        free_nats_mode: str = "step",
        group_mask_max: int = 1,
        overshoot_horizon: int = 0,
        overshoot_starts: int = 4,
        overshoot_kl_weight: float = 0.0,
        overshoot_obs_weight: float = 0.0,
        overshoot_free_nats: float = 0.0,
        predict_var: bool = False,
        var_weight: float = 1.0,
        score_mode: str = "mse",
        group_mask_sets: list[list[str]] | None = None,
        group_mask_set_prob: float = 0.0,
        group_mask_start_max: int = 0,
        consistency_sets: list[list[str]] | None = None,
        recon_mode_balance: bool = False,
    ):
        super().__init__(num_features, window_length)
        if free_nats_mode not in ("step", "dim"):
            raise ValueError(f"free_nats_mode must be 'step' or 'dim', got {free_nats_mode!r}")
        self.free_nats_mode = free_nats_mode
        if int(group_mask_max) < 1:
            raise ValueError("group_mask_max >= 1 required")
        self.group_mask_max = int(group_mask_max)
        self.overshoot_horizon = int(overshoot_horizon)
        self.overshoot_starts = int(overshoot_starts)
        self.overshoot_kl_weight = float(overshoot_kl_weight)
        self.overshoot_obs_weight = float(overshoot_obs_weight)
        self.overshoot_free_nats = float(overshoot_free_nats)
        if self.overshoot_horizon < 0 or not 0 <= self.overshoot_horizon < window_length:
            raise ValueError(f"overshoot_horizon must be in [0, {window_length})")
        if self.overshoot_horizon > 0 and (self.overshoot_kl_weight <= 0 and self.overshoot_obs_weight <= 0):
            raise ValueError("overshoot_horizon > 0 needs overshoot_kl_weight > 0 and/or overshoot_obs_weight > 0")
        if (self.overshoot_kl_weight > 0 or self.overshoot_obs_weight > 0) and self.overshoot_horizon == 0:
            raise ValueError("overshoot weights > 0 need overshoot_horizon > 0")
        if self.overshoot_obs_weight > 0 and not predict_increment:
            raise ValueError("overshoot_obs_weight needs predict_increment")
        if self.overshoot_horizon > 0 and self.overshoot_starts < 1:
            raise ValueError("overshoot_starts >= 1 required")
        self.stochastic_dim = stochastic_dim
        self.deterministic_dim = deterministic_dim
        self.kl_dyn_beta = kl_dyn_beta
        self.kl_rep_beta = kl_rep_beta
        self.free_nats = free_nats
        if not 0 <= score_burn_in < window_length:
            raise ValueError(
                f"score_burn_in={score_burn_in} must be in [0, {window_length})"
            )
        self.score_burn_in = score_burn_in
        self.predict_increment = bool(predict_increment)
        self.group_mask_prob = float(group_mask_prob)
        self.groups = [list(map(int, g)) for g in groups] if groups else None
        self.group_names = list(group_names) if group_names else None
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
            if self.group_mask_prob <= 0:
                raise ValueError("group_mask_sets needs group_mask_prob > 0")
            sm = torch.tensor([[g in s for g in self.group_names] for s in sets], dtype=torch.bool)
        else:
            sm = torch.zeros(0, len(self.group_names or []), dtype=torch.bool)
        self.register_buffer("group_set_masks", sm, persistent=False)
        # group_mask_start_max (opt-in, default 0): a masked sample hides its group(s) from a random step
        # s ~ U{0..group_mask_start_max} on instead of from the window start; before s the group is observed, at s the
        # increment base of the hidden channels starts at the last observation x[s-1] (as M2 of lever_metrics). Trains
        # the transition visible -> hidden (Vault modellumbau_rssm_mtsjepa, joint mask analysis 2026-11-20).
        self.group_mask_start_max = int(group_mask_start_max)
        if not 0 <= self.group_mask_start_max < window_length:
            raise ValueError(f"group_mask_start_max must be in [0, {window_length})")
        if self.group_mask_start_max > 0 and group_mask_prob <= 0:
            raise ValueError("group_mask_start_max > 0 needs group_mask_prob > 0")
        # consistency_sets (opt-in, scoring only): sets of groups scored by set_scores (the set hidden, error of its
        # prediction from the rest of the vehicle against the observation); no effect on training or anomaly_score
        self.consistency_sets = [list(c) for c in (consistency_sets or [])]
        # recon_mode_balance (opt-in, default off): with group masking the recon loss of the hidden channels is far
        # larger than that of the visible ones and carries 97-99 % of the gradient with the joint mask, so the one-step
        # prediction with all groups visible learns slowly (Vault modellumbau_rssm_mtsjepa, gate analysis 2026-11-20).
        # On: recon = sg(L) * (0.5 L_vis / sg(L_vis) + 0.5 L_hid / sg(L_hid)) with L the usual weighted MSE over all
        # elements and L_vis / L_hid the same over the visible / hidden elements: same value as before (balance with the
        # KL term unchanged), each mode gets half of the relative gradient. Batches without hidden elements: as before.
        self.recon_mode_balance = bool(recon_mode_balance)
        unknown = sorted({g for c in self.consistency_sets for g in c if g not in (self.group_names or [])})
        if unknown:
            raise ValueError(f"consistency_sets: unknown groups {unknown}")
        if self.group_mask_prob > 0 and not self.groups:
            raise ValueError("group_mask_prob > 0 needs groups (channel indices per group)")
        if hidden_base not in ("observed", "own", "own_input", "level"):
            raise ValueError(f"hidden_base must be 'observed', 'own', 'own_input' or 'level', got {hidden_base!r}")
        if hidden_base != "observed" and not (self.predict_increment and self.group_mask_prob > 0):
            raise ValueError(f"hidden_base {hidden_base!r} needs predict_increment and group_mask_prob > 0")
        self.hidden_base = hidden_base
        # window_relative: level channels relative to the window start (models/window_relative.py)
        self.win_rel = WindowRelative(window_relative) if window_relative else None
        # loss_weight: reconstruction weight per channel ("step": 1 / sd of the step^2, "step_sqrt": 1 / sd of
        # the step, both mean 1; computed in train.py on the training windows and stored as loss_weights)
        if loss_weight not in ("none", "step", "step_sqrt"):
            raise ValueError(f"loss_weight must be none, step or step_sqrt, got {loss_weight!r}")
        if (loss_weight != "none") != (loss_weights is not None):
            raise ValueError("loss_weight != 'none' needs loss_weights (set by train.py) and vice versa")
        self.loss_weight = loss_weight
        # loss_weight_floor (opt-in, default 0 = off): lower bound of the loss_weight weights as a fraction of their
        # median before the mean-1 normalization, applied in train.py (stored here for the record). Without it the
        # largest-step channels get tiny weights (network_down frame/payload_len 0.045 with step_sqrt), their decoder
        # is barely trained, and the own_input feedback of that hidden group becomes unstable beyond the data range
        # (rssm_wr_a_lw, Vault modellumbau_rssm_mtsjepa, 2026-11-14).
        if loss_weight_floor < 0 or (loss_weight_floor > 0 and loss_weight == "none"):
            raise ValueError("loss_weight_floor >= 0 required, and > 0 only with loss_weight != none")
        self.loss_weight_floor = float(loss_weight_floor)
        if loss_weights is not None:
            if len(loss_weights) != num_features:
                raise ValueError("loss_weights needs one weight per feature")
            self.register_buffer("loss_w", torch.tensor(loss_weights, dtype=torch.float32), persistent=False)
        else:
            self.loss_w = None
        # abs_inputs (opt-in): channels passed to the observation encoder as additional inputs with their value
        # BEFORE window_relative (absolute level, e.g. GLOBAL_POSITION_INT__relative_alt = height above home), so the
        # state can know the absolute level although all level channels are window-relative. Input only: not
        # predicted, not in loss, score or tune_metric. Hidden with the group of the channel. abs_input_idx is set by
        # train.py from the names (Vault modellumbau_rssm_mtsjepa, 2026-11-13).
        if bool(abs_inputs) != (abs_input_idx is not None):
            raise ValueError("abs_inputs needs abs_input_idx (set by train.py) and vice versa")
        self.abs_inputs = list(abs_inputs or [])
        if abs_input_idx is not None:
            if len(abs_input_idx) != len(self.abs_inputs) or not all(0 <= int(i) < num_features for i in abs_input_idx):
                raise ValueError("abs_input_idx: one valid feature index per abs_inputs entry")
            self.register_buffer("abs_idx", torch.tensor([int(i) for i in abs_input_idx], dtype=torch.long), persistent=False)
        else:
            self.abs_idx = None
        n_abs = len(self.abs_inputs)
        n_flags = len(self.groups) if self.group_mask_prob > 0 else 0
        self.n_groups = n_flags
        if n_flags:
            member = torch.zeros(n_flags, num_features)
            for gi, idx in enumerate(self.groups):
                member[gi, idx] = 1.0
            self.register_buffer("group_member", member, persistent=False)

        # Observation embedding
        self.obs_encoder = _mlp(num_features + n_abs + n_flags, hidden_dim, hidden_dim)

        # Deterministic path
        self.gru = nn.GRUCell(stochastic_dim, deterministic_dim)

        # Prior p(z_t | h_t) and posterior q(z_t | h_t, x_t): mean + log_std
        self.prior_net = _mlp(deterministic_dim, hidden_dim, 2 * stochastic_dim)
        self.posterior_net = _mlp(
            deterministic_dim + hidden_dim, hidden_dim, 2 * stochastic_dim
        )

        # Decoder D(h_t, z_t) -> x̂_t ("own_input": D(h_t, z_t, base of the hidden channels, group flags))
        extra = num_features + n_flags if hidden_base == "own_input" else 0
        self.decoder = _mlp(
            deterministic_dim + stochastic_dim + extra, hidden_dim, num_features
        )
        # "level": absolute level of the hidden channels L(h_t, z_t)
        self.level_head = (_mlp(deterministic_dim + stochastic_dim, hidden_dim, num_features)
                           if hidden_base == "level" else None)
        # predict_var (opt-in, step 1 of Vault modellumbau_rssm_mtsjepa 2026-11-18, uncertainty-aware score): a head
        # V(h_t, z_prior, decoder extra input) -> log-variance per channel of the one-step prior prediction. Trained with
        # the Gaussian NLL on DETACHED inputs and a DETACHED mean (var_weight), so the dynamics model and its mean
        # prediction learn exactly as without the head; only the score can change. score_mode "nll": anomaly_score =
        # mean of 0.5 * (log var + err / var) instead of the mean squared error ("mse", default). Hard-to-predict
        # channels (radio, pressure) then enter the score with their learned spread instead of dominating it.
        if score_mode not in ("mse", "nll"):
            raise ValueError(f"score_mode must be 'mse' or 'nll', got {score_mode!r}")
        if score_mode == "nll" and not predict_var:
            raise ValueError("score_mode 'nll' needs predict_var")
        if var_weight < 0:
            raise ValueError("var_weight >= 0 required")
        self.predict_var, self.var_weight, self.score_mode = bool(predict_var), float(var_weight), score_mode
        self.var_head = (_mlp(deterministic_dim + stochastic_dim + extra, hidden_dim, num_features)
                         if self.predict_var else None)

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor, gmask: torch.Tensor | None, xa: torch.Tensor | None = None,
                gstart: torch.Tensor | None = None) -> torch.Tensor:
        """Observation embedding; with group masking the hidden groups
        (gmask (B, G) bool) are zeroed and flagged. xa (B, T, n_abs): the abs_inputs
        (values before window_relative), zeroed with the group of their channel."""
        if self.abs_idx is not None and xa is None:
            raise ValueError("this model has abs_inputs: pass xa (the abs_inputs channels before window_relative)")
        parts = [x] if self.abs_idx is None else [x, xa]
        if not self.n_groups:
            return self.obs_encoder(torch.cat(parts, dim=-1) if len(parts) > 1 else x)
        if gmask is None:
            gmask = torch.zeros(x.shape[0], self.n_groups, dtype=torch.bool, device=x.device)
        g = gmask.float()
        keep = 1.0 - (g @ self.group_member).clamp(max=1.0)  # (B, F)
        if gstart is None:
            flags = g.unsqueeze(1).expand(-1, x.shape[1], -1)
            parts[0] = x * keep.unsqueeze(1)
            if self.abs_idx is not None:
                parts[1] = xa * keep[:, self.abs_idx].unsqueeze(1)
        else:  # hidden only from gstart (B,) on
            act = (torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= gstart.unsqueeze(1)).float()  # (B, T)
            keep_t = 1.0 - (1.0 - keep).unsqueeze(1) * act.unsqueeze(-1)  # (B, T, F)
            flags = g.unsqueeze(1) * act.unsqueeze(-1)
            parts[0] = x * keep_t
            if self.abs_idx is not None:
                parts[1] = xa * keep_t[:, :, self.abs_idx]
        return self.obs_encoder(torch.cat(parts + [flags], dim=-1))

    def sample_group_start(self, gmask: torch.Tensor | None) -> torch.Tensor | None:
        """Training: start step of the group mask per sample (group_mask_start_max > 0), else None."""
        if gmask is None or self.group_mask_start_max <= 0 or not self.training:
            return None
        return torch.randint(0, self.group_mask_start_max + 1, (gmask.shape[0],), device=gmask.device)

    def sample_group_mask(self, batch_size: int, device) -> torch.Tensor | None:
        """Training: per sample, with group_mask_prob, one random group hidden."""
        if not self.n_groups or not self.training:
            return None
        hide = torch.rand(batch_size, device=device) < self.group_mask_prob
        if self.group_mask_max > 1:  # group_mask_max: 1..max distinct groups per masked sample
            n = torch.randint(1, min(self.group_mask_max, self.n_groups) + 1, (batch_size,), device=device)
            rank = torch.rand(batch_size, self.n_groups, device=device).argsort(dim=1).argsort(dim=1)
            m = hide.unsqueeze(1) & (rank < n.unsqueeze(1))
        else:
            which = torch.randint(self.n_groups, (batch_size,), device=device)
            m = torch.zeros(batch_size, self.n_groups, dtype=torch.bool, device=device)
            m[hide, which[hide]] = True
        if self.group_mask_set_prob > 0:  # joint mask: replace by one of group_mask_sets
            use = hide & (torch.rand(batch_size, device=device) < self.group_mask_set_prob)
            k = torch.randint(self.group_set_masks.shape[0], (batch_size,), device=device)
            m = torch.where(use.unsqueeze(1), self.group_set_masks.to(device)[k], m)
        return m

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = stats.chunk(2, dim=-1)
        # Clamp for numerical stability. Floor raised from -5 to -3: at -5
        # (std ~0.0067) the prior's variance can collapse near zero, and
        # the KL term's 1/(2*var_p) factor then blows up on any batch
        # where the posterior mean drifts from the prior mean, spiking
        # the loss by several orders of magnitude (observed up to ~6e5).
        log_std = torch.clamp(log_std, -3.0, 2.0)
        return mean, log_std

    @staticmethod
    def _kl_diag_gaussian(
        mu_q: torch.Tensor, ls_q: torch.Tensor,
        mu_p: torch.Tensor, ls_p: torch.Tensor, per_dim: bool = False,
    ) -> torch.Tensor:
        """KL(q‖p) for diagonal Gaussians, summed over latent dim (per_dim: not summed)."""
        var_q, var_p = (2 * ls_q).exp(), (2 * ls_p).exp()
        kl = ls_p - ls_q + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
        return kl if per_dim else kl.sum(-1)

    def _rollout(
        self, x: torch.Tensor, use_posterior: bool,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_obs: torch.Tensor | None = None,
        gmask: torch.Tensor | None = None,
        return_latents: bool = False,
        gstart: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Unroll the RSSM over a window.

        Args:
            x: (B, T, F) observations.
            use_posterior: True for training (filtering), False to roll
                the prior only at the first step is *not* meant here —
                posterior is always used to update state; the flag
                controls whether decoded predictions use posterior
                (reconstruction) or prior (one-step prediction) latents.

        Returns dict with stacked tensors over T:
            recon_prior:  x̂ from prior latent  (B, T, F)
            recon_post:   x̂ from posterior latent (B, T, F)
            kl:           per-step KL (B, T)
            state:        (h, z) after the last step (for the next chunk)

        state / prev_obs (stateful use over a whole mission, opt-in): start
        from the given (h, z) instead of zeros, and with predict_increment
        use prev_obs (B, F), the last observation of the previous chunk,
        as x_{-1}. Both None = the window behaviour (unchanged).
        """
        xa = x[:, :, self.abs_idx] if self.abs_idx is not None else None  # absolute level, before window_relative
        if self.win_rel is not None:
            if state is not None or prev_obs is not None:
                raise NotImplementedError("window_relative is implemented for windows only (no carried state)")
            x = self.win_rel(x)
        B, T, _ = x.shape
        device = x.device
        if state is None:
            h = torch.zeros(B, self.deterministic_dim, device=device)
            z = torch.zeros(B, self.stochastic_dim, device=device)
        else:
            h, z = state

        recon_prior, recon_post = [], []
        logvars = []
        kls_dyn, kls_rep = [], []
        hs, zs = [], []  # return_latents: h_t and the posterior z_t per step (diagnostics, tune_metric.py)
        mqs, lqs = [], []  # return_latents: posterior mean / log std per step (overshooting targets)
        fb = self.free_nats_mode == "dim"
        kls_dyn_fb, kls_rep_fb = [], []  # free_nats_mode "dim": KL with free bits per dimension, summed
        if gstart is not None and (gmask is None or state is not None or prev_obs is not None):
            raise ValueError("gstart needs gmask and window use (no carried state)")
        obs_emb = self._encode(x, gmask, xa, gstart)  # (B, T, H)
        # hidden_base "own": the increment base of hidden channels is the model's own previous
        # (posterior) prediction, starting from 0 (= training mean), never the observation
        # "own_input": the same base, but the decoder sees it (and the group flags); "level": hidden channels
        # come from the level head (absolute, no base)
        hid = None
        if self.hidden_base != "observed" and gmask is not None and bool(gmask.any()):
            if self.hidden_base != "level" and (state is not None or prev_obs is not None):
                raise NotImplementedError(f"hidden_base {self.hidden_base!r} is implemented for windows only (no carried state)")
            hid = (gmask.float() @ self.group_member).clamp(max=1.0).bool()  # (B, F)
            base_h = torch.zeros_like(x[:, 0])
        if self.hidden_base == "own_input":
            flags_in = gmask.float() if gmask is not None else x.new_zeros(B, self.n_groups)

        for t in range(T):
            h = self.gru(z, h)
            # hidden channels at step t: all of hid without gstart, else only for samples with t >= gstart
            act = hid if (hid is None or gstart is None) else hid & (gstart <= t).unsqueeze(1)
            if self.hidden_base == "own_input":
                fl_t = flags_in if gstart is None else flags_in * (gstart <= t).unsqueeze(1).float()

            mu_p, ls_p = self._split_stats(self.prior_net(h))
            mu_q, ls_q = self._split_stats(
                self.posterior_net(torch.cat([h, obs_emb[:, t]], dim=-1))
            )

            if fb:
                kd = self._kl_diag_gaussian(mu_q.detach(), ls_q.detach(), mu_p, ls_p, per_dim=True)
                kr = self._kl_diag_gaussian(mu_q, ls_q, mu_p.detach(), ls_p.detach(), per_dim=True)
                kl_dyn, kl_rep = kd.sum(-1), kr.sum(-1)
                thr = self.free_nats / self.stochastic_dim
                kls_dyn_fb.append(torch.clamp(kd, min=thr).sum(-1)); kls_rep_fb.append(torch.clamp(kr, min=thr).sum(-1))
            else:
                kl_dyn = self._kl_diag_gaussian(
                    mu_q.detach(), ls_q.detach(), mu_p, ls_p
                )
                kl_rep = self._kl_diag_gaussian(
                    mu_q, ls_q, mu_p.detach(), ls_p.detach()
                )
            kls_dyn.append(kl_dyn)
            kls_rep.append(kl_rep)

            # Reparameterized samples (training); means at eval time.
            if self.training:
                z_prior = mu_p + ls_p.exp() * torch.randn_like(mu_p)
                z_post = mu_q + ls_q.exp() * torch.randn_like(mu_q)
            else:
                z_prior, z_post = mu_p, mu_q

            if self.hidden_base == "own_input":
                base_in = torch.where(act, base_h, torch.zeros_like(base_h)) if hid is not None else x.new_zeros(B, x.shape[-1])
                ex = torch.cat([base_in, fl_t], dim=-1)
                feat_prior = torch.cat([h, z_prior, ex], dim=-1)
                dec_prior = self.decoder(feat_prior)
                dec_post = self.decoder(torch.cat([h, z_post, ex], dim=-1))
            else:
                feat_prior = torch.cat([h, z_prior], dim=-1)
                dec_prior = self.decoder(feat_prior)
                dec_post = self.decoder(torch.cat([h, z_post], dim=-1))
            if self.var_head is not None:
                logvars.append(self.var_head(feat_prior.detach()).clamp(-10.0, 8.0))
            if self.predict_increment:
                if t > 0:
                    prev = x[:, t - 1]
                else:
                    prev = x[:, 0] if prev_obs is None else prev_obs
                if hid is not None and self.hidden_base != "level":
                    prev = torch.where(act, base_h, prev)
                dec_prior, dec_post = prev + dec_prior, prev + dec_post
                if hid is not None and self.hidden_base == "level":
                    dec_prior = torch.where(act, self.level_head(torch.cat([h, z_prior], dim=-1)), dec_prior)
                    dec_post = torch.where(act, self.level_head(torch.cat([h, z_post], dim=-1)), dec_post)
                elif hid is not None:
                    # before gstart the base follows the observation, so it starts at x[s-1] when the group is hidden
                    base_h = dec_post if gstart is None else torch.where(act, dec_post, x[:, t])
            recon_prior.append(dec_prior)
            recon_post.append(dec_post)

            z = z_post  # state update always uses the posterior (filtering)
            if return_latents:
                hs.append(h); zs.append(z_post); mqs.append(mu_q); lqs.append(ls_q)

        return {
            "recon_prior": torch.stack(recon_prior, dim=1),
            "recon_post": torch.stack(recon_post, dim=1),
            "kl_dyn": torch.stack(kls_dyn, dim=1),
            "kl_rep": torch.stack(kls_rep, dim=1),
            "state": (h, z),
            "x": x,  # the window as the model sees it (window_relative applied); residuals refer to it
            **({"logvar_prior": torch.stack(logvars, dim=1)} if logvars else {}),
            **({"h": torch.stack(hs, dim=1), "z": torch.stack(zs, dim=1),
                "mu_q": torch.stack(mqs, dim=1), "ls_q": torch.stack(lqs, dim=1)} if return_latents else {}),
            **({"kl_dyn_fb": torch.stack(kls_dyn_fb, dim=1), "kl_rep_fb": torch.stack(kls_rep_fb, dim=1)} if fb else {}),
        }

    def _overshoot(self, out: dict, starts: torch.Tensor, horizon: int, sample: bool) -> dict[str, torch.Tensor]:
        """Prior rollout of `horizon` steps from the filter state at each start step s (levers b1/b2, see module
        docstring). out: _rollout(..., return_latents=True). starts: (K,) long, s + horizon < T. sample: z from the
        prior drawn (training) or its mean. Returns kl (B, K, D) = KL(sg(q_{s+d}) || p_{s+d|s}) and pred (B, K, D, F)
        = accumulated prediction x_s + sum of the decoded increments (predict_increment only; decoder extra input as
        unmasked: base 0, flags 0)."""
        H, Z, x = out["h"], out["z"], out["x"]
        B, T, F_ = x.shape; K = starts.numel()
        h = H[:, starts].reshape(B * K, -1); z = Z[:, starts].reshape(B * K, -1)
        prev = x[:, starts].reshape(B * K, F_) if self.predict_increment else None
        ex = x.new_zeros(B * K, F_ + self.n_groups) if self.hidden_base == "own_input" else None
        kls, preds = [], []
        for d in range(1, horizon + 1):
            h = self.gru(z, h)
            mu_p, ls_p = self._split_stats(self.prior_net(h))
            z = mu_p + ls_p.exp() * torch.randn_like(mu_p) if sample else mu_p
            tq = starts + d
            mq = out["mu_q"][:, tq].reshape(B * K, -1).detach(); lq = out["ls_q"][:, tq].reshape(B * K, -1).detach()
            kls.append(self._kl_diag_gaussian(mq, lq, mu_p, ls_p).view(B, K))
            if prev is not None:
                dec = self.decoder(torch.cat([h, z, ex], dim=-1) if ex is not None else torch.cat([h, z], dim=-1))
                prev = prev + dec; preds.append(prev.view(B, K, F_))
        return {"kl": torch.stack(kls, dim=2), **({"pred": torch.stack(preds, dim=2)} if preds else {})}

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _recon_loss(self, pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.loss_w is None:
            return F.mse_loss(pred, x)
        return ((pred - x) ** 2 * self.loss_w).mean()

    def _recon_loss_balanced(self, pred, x, gmask, gstart, total):
        """recon_mode_balance: per-mode normalized recon loss (see __init__); total = the usual _recon_loss."""
        se = (pred - x) ** 2 * (self.loss_w if self.loss_w is not None else 1.0)
        hid = (gmask.float() @ self.group_member).clamp(max=1.0).bool().unsqueeze(1).expand_as(se)  # (B, T, F)
        if gstart is not None:
            hid = hid & (torch.arange(se.shape[1], device=se.device).view(1, -1, 1) >= gstart.view(-1, 1, 1))
        if not bool(hid.any()) or bool(hid.all()):
            return total
        lv, lh = se[~hid].mean(), se[hid].mean()
        return total.detach() * (0.5 * lv / lv.detach().clamp_min(1e-12) + 0.5 * lh / lh.detach().clamp_min(1e-12))

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        gmask = self.sample_group_mask(batch.shape[0], batch.device)
        gstart = self.sample_group_start(gmask)
        self.last_gmask = gmask  # diagnostics (train.py training.spike_dump_threshold)
        os_on = self.overshoot_horizon > 0
        out = self._rollout(batch, use_posterior=True, gmask=gmask, return_latents=os_on, gstart=gstart)
        recon_loss = self._recon_loss(out["recon_post"], out["x"])
        if self.recon_mode_balance and gmask is not None and bool(gmask.any()):
            recon_loss = self._recon_loss_balanced(out["recon_post"], out["x"], gmask, gstart, recon_loss)
        # Free nats applied per element, before averaging: each (batch,
        # timestep) KL value is floored at free_nats individually, so
        # elements already above the threshold keep their own gradient
        # instead of being averaged together with below-threshold ones
        # and zeroed out as a whole once the batch mean dips below it.
        if self.free_nats_mode == "dim":  # free bits per latent dimension (lever a2)
            kl_dyn = out["kl_dyn_fb"].mean(); kl_rep = out["kl_rep_fb"].mean()
        else:
            kl_dyn = torch.clamp(out["kl_dyn"], min=self.free_nats).mean()
            kl_rep = torch.clamp(out["kl_rep"], min=self.free_nats).mean()
        loss = recon_loss + self.kl_dyn_beta * kl_dyn + self.kl_rep_beta * kl_rep
        res = {
            "loss": loss,
            "recon": recon_loss.detach(),
            "kl_dyn": kl_dyn.detach(),
            "kl_rep": kl_rep.detach(),
        }
        if self.var_head is not None:  # NLL of the variance head, mean and features detached (no effect on the model)
            nll = self.nll_terms(out, detach_mean=True)[:, self.score_burn_in:].mean()
            loss = loss + self.var_weight * nll; res["loss"] = loss; res["var_nll"] = nll.detach()
        if os_on:  # overshooting (levers b1/b2)
            T = out["x"].shape[1]; D = self.overshoot_horizon
            starts = torch.randperm(T - D, device=batch.device)[: self.overshoot_starts]
            os_ = self._overshoot(out, starts, D, sample=self.training)
            if self.overshoot_kl_weight > 0:
                os_kl = torch.clamp(os_["kl"], min=self.overshoot_free_nats).mean()
                loss = loss + self.overshoot_kl_weight * os_kl; res["os_kl"] = os_kl.detach()
            if self.overshoot_obs_weight > 0:
                tgt = out["x"][:, starts.unsqueeze(1) + torch.arange(1, D + 1, device=batch.device)]  # (B, K, D, F)
                err = (os_["pred"] - tgt) ** 2
                os_obs = (err * self.loss_w).mean() if self.loss_w is not None else err.mean()
                loss = loss + self.overshoot_obs_weight * os_obs; res["os_obs"] = os_obs.detach()
            res["loss"] = loss
        return res

    def training_step_stateful(
        self, batch: torch.Tensor, state, prev_obs, reset: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], tuple, torch.Tensor]:
        """Training step on one chunk of a mission (truncated BPTT, opt-in
        `training.stateful`). state = (h, z) carried from the previous chunk
        of the same mission, prev_obs its last observation; rows with
        reset = True start a new mission (zero state, x_{-1} := x_0).
        Gradients stop at the chunk boundary (state detached). Loss as in
        training_step. Returns (outputs, new detached state, last obs)."""
        if self.hidden_base in ("own", "own_input") or self.win_rel is not None:
            raise NotImplementedError("hidden_base 'own'/'own_input' and window_relative are implemented for windows only")
        if self.overshoot_horizon > 0 or self.free_nats_mode != "step" or self.predict_var:
            raise NotImplementedError("overshooting, free_nats_mode 'dim' and predict_var are implemented for windows only")
        h, z = state
        keep = (~reset).float().unsqueeze(-1)
        h, z = h * keep, z * keep
        prev = torch.where(reset.unsqueeze(-1), batch[:, 0], prev_obs)
        out = self._rollout(batch, use_posterior=True, state=(h.detach(), z.detach()), prev_obs=prev,
                            gmask=self.sample_group_mask(batch.shape[0], batch.device))
        recon_loss = self._recon_loss(out["recon_post"], batch)
        kl_dyn = torch.clamp(out["kl_dyn"], min=self.free_nats).mean()
        kl_rep = torch.clamp(out["kl_rep"], min=self.free_nats).mean()
        loss = recon_loss + self.kl_dyn_beta * kl_dyn + self.kl_rep_beta * kl_rep
        h1, z1 = out["state"]
        res = {"loss": loss, "recon": recon_loss.detach(), "kl_dyn": kl_dyn.detach(), "kl_rep": kl_rep.detach(),
               "kl_dyn_step": out["kl_dyn"].detach().mean(dim=1), "h_norm": h1.detach().norm(dim=-1)}
        return res, (h1.detach(), z1.detach()), batch[:, -1].detach()

    def nll_terms(self, out: dict, detach_mean: bool = False) -> torch.Tensor:
        """Gaussian NLL per element of the one-step prior prediction, (B, T, F): 0.5 * (log var + err / var)."""
        mu = out["recon_prior"].detach() if detach_mean else out["recon_prior"]
        lv = out["logvar_prior"]
        return 0.5 * (lv + (mu - out["x"]) ** 2 * torch.exp(-lv))

    @torch.no_grad()
    def element_scores(self, batch: torch.Tensor) -> torch.Tensor:
        """Per-element score (B, T, F) of anomaly_score (squared error or NLL, by score_mode), for group tools."""
        self.eval()
        out = self._rollout(batch, use_posterior=False)
        return self.nll_terms(out) if self.score_mode == "nll" else (out["recon_prior"] - out["x"]) ** 2

    @torch.no_grad()
    def set_scores(self, batch: torch.Tensor, sets: list[list[str]] | None = None) -> torch.Tensor:
        """Masked consistency score (B, S): per set (default consistency_sets) the set hidden over the whole window,
        mean squared error of its posterior prediction (from the rest of the vehicle, hidden_base own_input) against
        the observation on the set's channels after the burn-in. Opt-in, scoring only (Vault modellumbau_rssm_mtsjepa,
        joint mask analysis 2026-11-20)."""
        self.eval()
        sets = self.consistency_sets if sets is None else [list(c) for c in sets]
        if not sets or not self.group_names:
            raise ValueError("set_scores needs sets of group names")
        cols = []
        for c in sets:
            gm = torch.zeros(batch.shape[0], self.n_groups, dtype=torch.bool, device=batch.device)
            for g in c:
                gm[:, self.group_names.index(g)] = True
            ch = (gm[:1].float() @ self.group_member).clamp(max=1.0).bool()[0]
            out = self._rollout(batch, use_posterior=True, gmask=gm)
            err = (out["recon_post"] - out["x"]) ** 2
            cols.append(err[:, self.score_burn_in:][:, :, ch].mean(dim=(1, 2)))
        return torch.stack(cols, dim=1)

    @torch.no_grad()
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        self.eval()
        if self.score_mode == "nll":
            return self.element_scores(batch)[:, self.score_burn_in:].mean(dim=(1, 2))
        out = self._rollout(batch, use_posterior=False)
        # One-step prediction error from the prior: how well does the
        # learned dynamics model predict the next observation?
        err = (out["recon_prior"] - out["x"]) ** 2  # (B, T, F)
        return err[:, self.score_burn_in:].mean(dim=(1, 2))