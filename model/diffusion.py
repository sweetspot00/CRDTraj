"""
DDPM noise schedule, forward/reverse process, and Tweedie formula.

Equations referenced from the CRDTraj paper:
  Eq. (2)  forward process: q(τ_t | τ_0) = N(τ_t; √ᾱ_t τ_0, (1−ᾱ_t)I)
  Eq. (4)  reverse mean:    μ_θ = (1/√α_t)(τ_t − √(1−α_t)/√(1−ᾱ_t) ε_θ)
  Eq. (5)  Tweedie:         τ̂_0 = (τ_t − √(1−ᾱ_t) ε_θ) / √ᾱ_t
"""

import math
import torch
import torch.nn as nn


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    ᾱ_t = cos²((t/T + s)/(1 + s) · π/2)
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_bar = f / f[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(0.0, 0.999).float()


class DiffusionSchedule(nn.Module):
    """
    Pre-computes and registers all diffusion schedule buffers.

    Buffers (all shape [T]):
      betas, alphas, alphas_bar, sqrt_alphas_bar, sqrt_one_minus_alphas_bar,
      posterior_mean_coef1, posterior_mean_coef2, sigmas
    """

    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        super().__init__()
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        self.T = T
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())
        # Posterior variance σ_t² = β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.register_buffer("sigmas", posterior_variance.sqrt())
        # Reverse mean coefficients (Eq. 4)
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * alphas_bar_prev.sqrt()) / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_bar_prev) * alphas.sqrt()) / (1.0 - alphas_bar),
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Gather schedule values at timesteps t and broadcast to shape."""
        out = a.gather(0, t)
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward noising: Eq. (2)
        τ_t = √ᾱ_t · τ_0 + √(1−ᾱ_t) · ε,  ε ~ N(0, I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Tweedie formula: Eq. (5)
        τ̂_0 = (τ_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
        """
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, xt.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, xt.shape)
        return (xt - sqrt_1mab * eps) / sqrt_ab

    def p_mean(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Reverse mean: Eq. (4)
        μ_θ = (1/√α_t)(τ_t − β_t/√(1−ᾱ_t) · ε_θ)
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, xt.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, xt.shape)
        x0 = self.predict_x0(xt, t, eps)
        return coef1 * x0 + coef2 * xt

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio: SNR(t) = ᾱ_t / (1 − ᾱ_t).
        Returns scalar tensor per timestep.
        """
        ab = self.alphas_bar[t]
        return ab / (1.0 - ab)

    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single reverse step (Eq. 28):
        τ_{t-1} = μ_θ(τ_t, t) + guidance + σ_t · z,  z ~ N(0, I)

        guidance: pre-computed ∑_k g_k · ∇_{τ_t} r̂_k  (same shape as xt)
        """
        mean = self.p_mean(xt, t, eps)
        if guidance is not None:
            mean = mean + guidance
        sigma = self._extract(self.sigmas, t, xt.shape)
        noise = torch.randn_like(xt)
        # No noise at t=0
        mask = (t > 0).float()
        while mask.ndim < xt.ndim:
            mask = mask.unsqueeze(-1)
        return mean + sigma * noise * mask
