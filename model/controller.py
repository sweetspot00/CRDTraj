"""
Adaptive Guidance Controller for CRDTraj — Eqs. (28-31).

AdaptiveController (w_ψ) — Eq. (29):
  Input:  [SNR(t), e_ctx, R̂_t, [‖∇r̂_k‖]_{k=1}^6]
  Output: [g₁(t), ..., g₆(t)] = g_max · sigmoid(MLP_ψ(...))

REINFORCE training loss — Eq. (30-31).
"""

import torch
import torch.nn as nn


class AdaptiveController(nn.Module):
    """
    Lightweight 3-layer MLP that outputs per-component guidance gates g_k(t).

    Architecture (Eq. 29):
      input_dim → 64 → 64 → 6
      output = g_max · sigmoid(logits)

    Input features (concatenated):
      - SNR(t):        scalar signal-to-noise ratio                  dim 1
      - e_ctx:         context embedding (d-dimensional)             dim d
      - R̂_t:          scalar total reward estimate                  dim 1
      - ‖∇r̂_k‖:       gradient norms for each of 6 sub-rewards     dim 6
      Total input dim: 8 + d

    Args:
        d:      context embedding dimension (from backbone)
        g_max:  maximum gate value (scales the output range)
    """

    def __init__(self, d: int, g_max: float = 1.0):
        super().__init__()
        self.g_max = g_max
        input_dim = 1 + d + 1 + 6  # SNR + e_ctx + R_hat + grad_norms

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )

    def forward(
        self,
        snr: torch.Tensor,
        e_ctx: torch.Tensor,
        R_hat: torch.Tensor,
        grad_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gates g_k(t) for all 6 sub-reward components.

        Args:
            snr:        (B,)   SNR(t) = ᾱ_t / (1 − ᾱ_t)
            e_ctx:      (B, d) context embedding (pooled from backbone)
            R_hat:      (B,)   total reward estimate from reward head
            grad_norms: (B, 6) ‖∇_{τ_t} r̂_k‖ for each component
        Returns:
            gates: (B, 6) ∈ [0, g_max]
        """
        x = torch.cat([
            snr.unsqueeze(-1),       # (B, 1)
            e_ctx,                   # (B, d)
            R_hat.unsqueeze(-1),     # (B, 1)
            grad_norms,              # (B, 6)
        ], dim=-1)                   # (B, 1+d+1+6)

        logits = self.mlp(x)         # (B, 6)
        return self.g_max * torch.sigmoid(logits)

    def log_prob(self, gates: torch.Tensor, snr, e_ctx, R_hat, grad_norms) -> torch.Tensor:
        """
        Log-probability of the gate values under the current policy for REINFORCE.
        We model each gate dimension as Gaussian with fixed small variance around the
        sigmoid mean (continuous policy approximation).

        Returns:
            (B,) summed log-prob over all 6 gates
        """
        mean = self.forward(snr, e_ctx, R_hat, grad_norms)  # (B, 6)
        # Use Normal distribution around the mean; std = 0.1 (fixed)
        dist = torch.distributions.Normal(mean, 0.1)
        return dist.log_prob(gates.clamp(1e-6, self.g_max - 1e-6)).sum(dim=-1)


def reinforce_loss(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    gates: torch.Tensor,
    baseline: float = 0.0,
    beta: float = 1e-3,
) -> torch.Tensor:
    """
    Controller REINFORCE training loss — Eqs. (30-31).

    L_ctrl = −E[(R − R̄) · log π_ψ(g)] + β ∑_{t,k} g_k(t)²

    Args:
        rewards:   (B,) final trajectory rewards R(τ₀^(b))
        log_probs: (B,) log π_ψ(g^(b)) summed over timesteps and components
        gates:     (B, T, 6) or (B, 6) gate values used during rollout
        baseline:  running average reward R̄ for variance reduction
        beta:      regularisation coefficient
    Returns:
        scalar loss
    """
    advantage = rewards - baseline                           # (B,)
    policy_loss = -(advantage.detach() * log_probs).mean()   # Eq. (31)
    reg_loss = beta * (gates ** 2).sum(dim=-1).mean()        # L2 gate regularisation
    return policy_loss + reg_loss
