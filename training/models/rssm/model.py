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
from the learned dynamics is the anomaly signal).

Gaussian latents are used instead of DreamerV3's categorical latents:
they are ONNX-friendly and sufficient for low-dimensional telemetry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import AnomalyDetectionModel


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
    ):
        super().__init__(num_features, window_length)
        self.stochastic_dim = stochastic_dim
        self.deterministic_dim = deterministic_dim
        self.kl_dyn_beta = kl_dyn_beta
        self.kl_rep_beta = kl_rep_beta
        self.free_nats = free_nats

        # Observation embedding
        self.obs_encoder = _mlp(num_features, hidden_dim, hidden_dim)

        # Deterministic path
        self.gru = nn.GRUCell(stochastic_dim, deterministic_dim)

        # Prior p(z_t | h_t) and posterior q(z_t | h_t, x_t): mean + log_std
        self.prior_net = _mlp(deterministic_dim, hidden_dim, 2 * stochastic_dim)
        self.posterior_net = _mlp(
            deterministic_dim + hidden_dim, hidden_dim, 2 * stochastic_dim
        )

        # Decoder D(h_t, z_t) -> x̂_t
        self.decoder = _mlp(
            deterministic_dim + stochastic_dim, hidden_dim, num_features
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = stats.chunk(2, dim=-1)
        # Clamp for numerical stability
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    @staticmethod
    def _kl_diag_gaussian(
        mu_q: torch.Tensor, ls_q: torch.Tensor,
        mu_p: torch.Tensor, ls_p: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q‖p) for diagonal Gaussians, summed over latent dim."""
        var_q, var_p = (2 * ls_q).exp(), (2 * ls_p).exp()
        kl = ls_p - ls_q + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
        return kl.sum(-1)

    def _rollout(
        self, x: torch.Tensor, use_posterior: bool
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
        """
        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.deterministic_dim, device=device)
        z = torch.zeros(B, self.stochastic_dim, device=device)

        recon_prior, recon_post = [], []
        kls_dyn, kls_rep = [], []
        obs_emb = self.obs_encoder(x)  # (B, T, H)

        for t in range(T):
            h = self.gru(z, h)

            mu_p, ls_p = self._split_stats(self.prior_net(h))
            mu_q, ls_q = self._split_stats(
                self.posterior_net(torch.cat([h, obs_emb[:, t]], dim=-1))
            )

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

            recon_prior.append(self.decoder(torch.cat([h, z_prior], dim=-1)))
            recon_post.append(self.decoder(torch.cat([h, z_post], dim=-1)))

            z = z_post  # state update always uses the posterior (filtering)

        return {
            "recon_prior": torch.stack(recon_prior, dim=1),
            "recon_post": torch.stack(recon_post, dim=1),
            "kl_dyn": torch.stack(kls_dyn, dim=1),
            "kl_rep": torch.stack(kls_rep, dim=1),
        }

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self._rollout(batch, use_posterior=True)
        recon_loss = F.mse_loss(out["recon_post"], batch)
        # Free nats applied per term (clamp keeps gradient at zero only
        # below the threshold of the respective term).
        kl_dyn = torch.clamp(out["kl_dyn"].mean(), min=self.free_nats)
        kl_rep = torch.clamp(out["kl_rep"].mean(), min=self.free_nats)
        loss = recon_loss + self.kl_dyn_beta * kl_dyn + self.kl_rep_beta * kl_rep
        return {
            "loss": loss,
            "recon": recon_loss.detach(),
            "kl_dyn": kl_dyn.detach(),
            "kl_rep": kl_rep.detach(),
        }

    @torch.no_grad()
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self._rollout(batch, use_posterior=False)
        # One-step prediction error from the prior: how well does the
        # learned dynamics model predict the next observation?
        err = (out["recon_prior"] - batch) ** 2  # (B, T, F)
        return err.mean(dim=(1, 2))