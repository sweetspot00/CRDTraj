"""
Output heads for CRDTraj:
  DenoisingHead — Eq. (14)  MLP(d → T*2) per agent → reshape to (T, 2)
  RewardHead    — Eq. (15)  MLP([z̄ ∥ Pool(τ̂₀)] → 6)
"""

import torch
import torch.nn as nn


def _make_mlp(dims: list[int], act: type = nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class DenoisingHead(nn.Module):
    """
    Predicts the noise ε_θ^i for each agent — Eq. (14).

    ε_θ^i = MLP_denoise(z_i) ∈ R^{T×2}
    Full output: ε_θ ∈ R^{N×T×2}

    Args:
        d: model (token) dimension
        T: trajectory length (number of timesteps)
    """

    def __init__(self, d: int, T: int):
        super().__init__()
        self.T = T
        self.mlp = _make_mlp([d, d, T * 2])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, d)  agent representations
        Returns:
            eps: (B, N, T, 2)  predicted noise
        """
        B, N, d = z.shape
        out = self.mlp(z)                       # (B, N, T*2)
        return out.reshape(B, N, self.T, 2)     # (B, N, T, 2)


class RewardHead(nn.Module):
    """
    Predicts 6 sub-reward scalars — Eq. (15).

    Input:
      z̄      = mean-pooled agent representation:   R^d
      Pool(τ̂₀) = trajectory statistics from τ̂₀:   R^{pool_dim}

    Concatenated input → MLP → R^6 (passed through sigmoid to [0, 1]).

    Trajectory statistics extracted from τ̂₀ (B, N, T, 2):
      - mean speed per agent → mean over agents         (1,)
      - std of speeds                                   (1,)
      - spatial spread (std of final positions)         (2,)
      - mean direction vector (normalised displacement) (2,)
    Total pool_dim = 6.

    Args:
        d:        model dimension
        pool_dim: dimension of τ̂₀ statistics (default 6)
    """

    POOL_DIM = 6

    def __init__(self, d: int, pool_dim: int = 6):
        super().__init__()
        in_dim = d + pool_dim
        self.mlp = _make_mlp([in_dim, d, 6])

    @staticmethod
    def pool_trajectory(tau0_hat: torch.Tensor) -> torch.Tensor:
        """
        Extract trajectory-level statistics from τ̂₀.

        Args:
            tau0_hat: (B, N, T, 2)
        Returns:
            stats: (B, 6)
        """
        # Per-step displacements → speeds  shape (B, N, T-1)
        diffs = tau0_hat[:, :, 1:, :] - tau0_hat[:, :, :-1, :]  # (B, N, T-1, 2)
        speeds = diffs.norm(dim=-1)                               # (B, N, T-1)
        per_agent_speed = speeds.mean(dim=-1)                     # (B, N)

        mean_speed = per_agent_speed.mean(dim=1, keepdim=True)    # (B, 1)
        std_speed  = per_agent_speed.std(dim=1, keepdim=True).clamp(min=0)  # (B, 1)

        # Spatial spread of final positions
        final_pos = tau0_hat[:, :, -1, :]                        # (B, N, 2)
        spread = final_pos.std(dim=1)                             # (B, 2)

        # Mean normalised displacement (direction)
        disp = tau0_hat[:, :, -1, :] - tau0_hat[:, :, 0, :]     # (B, N, 2)
        disp_norm = disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        unit_disp = (disp / disp_norm).mean(dim=1)               # (B, 2)

        return torch.cat([mean_speed, std_speed, spread, unit_disp], dim=-1)  # (B, 6)

    def forward(self, z: torch.Tensor, tau0_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:        (B, N, d)    agent representations from backbone
            tau0_hat: (B, N, T, 2) Tweedie-estimated clean trajectories
        Returns:
            reward_preds: (B, 6)  predicted sub-rewards in [0, 1]
        """
        z_bar = z.mean(dim=1)                      # (B, d)
        pool = self.pool_trajectory(tau0_hat)       # (B, 6)
        x = torch.cat([z_bar, pool], dim=-1)       # (B, d+6)
        logits = self.mlp(x)                        # (B, 6)
        return torch.sigmoid(logits)                # (B, 6)  ∈ [0, 1]
