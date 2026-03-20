"""
Reward functions for CRDTraj — Eqs. (16–25).

All functions operate on clean or estimated clean trajectories τ₀ ∈ R^{N×T×2}
and return scalar tensors in [0, 1].

API convention:
  All public reward functions accept a batch dimension B:
    tau0: (B, N, T, 2) — positions in metres
    Δt:   float         — time between steps in seconds
  and return (B,) tensors.

band_reward(x, a, b, sigma) — Eq. (16) — the shared building block.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Eq. (16) — Band-Shaped Reward Template
# ---------------------------------------------------------------------------

def band_reward(x: torch.Tensor, a: float, b: float, sigma: float) -> torch.Tensor:
    """
    Band(x; a, b, σ) = exp(−max(0, a−x)²/(2σ²)) · exp(−max(0, x−b)²/(2σ²))

    Equals 1.0 when x ∈ [a, b]; smooth Gaussian decay outside.
    Works element-wise on any shape tensor.
    """
    left  = torch.clamp(a - x, min=0.0)   # penalty for x < a
    right = torch.clamp(x - b, min=0.0)   # penalty for x > b
    return torch.exp(-left**2 / (2 * sigma**2)) * torch.exp(-right**2 / (2 * sigma**2))


# ---------------------------------------------------------------------------
# Eq. (17) — r1: Speed Compliance Reward
# ---------------------------------------------------------------------------

def speed_reward(
    tau0: torch.Tensor,
    v_min: float,
    v_max: float,
    sigma_v: float = 0.5,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    v̄^i = mean speed of agent i over T-1 steps.
    r_speed = (1/N) ∑_i Band(v̄^i; v_min, v_max, σ_v)

    Args:
        tau0:  (B, N, T, 2) clean trajectories
        v_min, v_max: desired speed band (m/s)
        sigma_v: Gaussian decay width
        dt: time step (s)
    Returns:
        (B,)
    """
    diffs  = tau0[:, :, 1:, :] - tau0[:, :, :-1, :]   # (B, N, T-1, 2)
    speeds = diffs.norm(dim=-1) / dt                    # (B, N, T-1)  m/s
    v_bar  = speeds.mean(dim=-1)                        # (B, N)
    per_agent = band_reward(v_bar, v_min, v_max, sigma_v)  # (B, N)
    return per_agent.mean(dim=1)                         # (B,)


# ---------------------------------------------------------------------------
# Eqs. (18-20) — r2/r3: Collision Rate Rewards
# ---------------------------------------------------------------------------

def _soft_collision(tau0: torch.Tensor, d_col: float, xi: float) -> torch.Tensor:
    """
    col_soft(i, j, t) = sigmoid((d_col − ‖pos_t^i − pos_t^j‖) / ξ)

    Returns (B, N, N, T) pairwise soft collision indicators.
    """
    pos = tau0  # (B, N, T, 2)
    B, N, T, _ = pos.shape
    pi = pos.unsqueeze(2).expand(B, N, N, T, 2)   # (B, N, N, T, 2)
    pj = pos.unsqueeze(1).expand(B, N, N, T, 2)
    dist = (pi - pj).norm(dim=-1)                  # (B, N, N, T)
    return torch.sigmoid((d_col - dist) / xi)       # (B, N, N, T)


def collision_reward(
    tau0: torch.Tensor,
    c_min: float,
    c_max: float,
    sigma_c: float = 0.1,
    d_col: float = 1.0,
    xi: float = 0.1,
    phase: str = "early",
) -> torch.Tensor:
    """
    r_col-e / r_col-l — Eqs. (19-20).

    c̄ = (2 / (N(N-1)·⌊T/2⌋)) ∑_t ∑_{i<j} col_soft(i,j,t)
    computed over first (phase='early') or second (phase='late') half.

    Args:
        tau0:  (B, N, T, 2)
        c_min, c_max: desired collision rate band
        sigma_c: Gaussian decay width
        d_col:  collision distance threshold (m)
        xi:     soft indicator sharpness
        phase:  'early' (first ⌊T/2⌋ steps) or 'late' (remaining steps)
    Returns:
        (B,)
    """
    B, N, T, _ = tau0.shape
    half = T // 2

    col = _soft_collision(tau0, d_col, xi)  # (B, N, N, T)

    if phase == "early":
        col = col[:, :, :, :half]           # (B, N, N, ⌊T/2⌋)
        n_steps = half
    else:
        col = col[:, :, :, half:]           # (B, N, N, T-⌊T/2⌋)
        n_steps = T - half

    # Upper-triangle mask to count each pair once
    mask = torch.triu(torch.ones(N, N, device=tau0.device, dtype=torch.bool), diagonal=1)
    # (N, N) → broadcast to (B, N, N, n_steps)
    col = col * mask.unsqueeze(0).unsqueeze(-1)

    n_pairs = N * (N - 1) / 2
    c_bar = col.sum(dim=(1, 2, 3)) / (n_pairs * n_steps + 1e-8)  # (B,)
    return band_reward(c_bar, c_min, c_max, sigma_c)               # (B,)


def collision_reward_early(tau0, c_min, c_max, sigma_c=0.1, d_col=1.0, xi=0.1):
    return collision_reward(tau0, c_min, c_max, sigma_c, d_col, xi, phase="early")


def collision_reward_late(tau0, c_min, c_max, sigma_c=0.1, d_col=1.0, xi=0.1):
    return collision_reward(tau0, c_min, c_max, sigma_c, d_col, xi, phase="late")


# ---------------------------------------------------------------------------
# Eqs. (21-22) — r4: Event Directionality Reward
# ---------------------------------------------------------------------------

def event_reward(
    tau0: torch.Tensor,
    event_center: torch.Tensor,
    direction: str,
    rho_min: float,
    rho_max: float,
    sigma_rho: float = 0.1,
    xi_e: float = 0.5,
) -> torch.Tensor:
    """
    Δd^i = ‖pos_T^i − e‖ − ‖pos_0^i − e‖  (positive = moved away from e)

    r_dir^i = sigmoid(−Δd^i/ξ_e)  if direction='toward'
              sigmoid(+Δd^i/ξ_e)  if direction='away'

    ρ = (1/N) ∑_i r_dir^i
    r_event = Band(ρ; ρ_min, ρ_max, σ_ρ)

    Args:
        tau0:         (B, N, T, 2)
        event_center: (B, 2) or (2,) event location
        direction:    'toward' or 'away'
        rho_min/max:  desired relevance band
        sigma_rho:    Gaussian decay width
        xi_e:         sigmoid sharpness
    Returns:
        (B,)
    """
    if event_center.ndim == 1:
        event_center = event_center.unsqueeze(0)  # (1, 2)
    e = event_center.unsqueeze(1)               # (B, 1, 2) or (1, 1, 2)

    pos0 = tau0[:, :, 0, :]                     # (B, N, 2)
    posT = tau0[:, :, -1, :]                    # (B, N, 2)

    d_start = (pos0 - e).norm(dim=-1)           # (B, N)
    d_end   = (posT - e).norm(dim=-1)           # (B, N)
    delta_d = d_end - d_start                   # positive = moved away

    if direction == "toward":
        r_dir = torch.sigmoid(-delta_d / xi_e)
    else:
        r_dir = torch.sigmoid(delta_d / xi_e)

    rho = r_dir.mean(dim=1)                     # (B,)
    return band_reward(rho, rho_min, rho_max, sigma_rho)


# ---------------------------------------------------------------------------
# Eq. (23) — r5: Goal Achievement Reward
# ---------------------------------------------------------------------------

def goal_reward(
    tau0: torch.Tensor,
    goals: torch.Tensor | None,
    d_goal: bool,
    sigma_g: float = 1.0,
) -> torch.Tensor:
    """
    r_goal = (1/N) ∑_i exp(−‖pos_T^i − g^i‖²/(2σ_g²))  if d_goal=True
             1.0                                          otherwise

    Args:
        tau0:  (B, N, T, 2)
        goals: (B, N, 2) per-agent goal positions (required if d_goal=True)
        d_goal: whether goal achievement is specified
        sigma_g: goal proximity bandwidth (m)
    Returns:
        (B,)
    """
    if not d_goal:
        return torch.ones(tau0.shape[0], device=tau0.device)

    assert goals is not None, "goals must be provided when d_goal=True"
    posT = tau0[:, :, -1, :]                            # (B, N, 2)
    dist_sq = ((posT - goals) ** 2).sum(dim=-1)         # (B, N)
    per_agent = torch.exp(-dist_sq / (2 * sigma_g**2))  # (B, N)
    return per_agent.mean(dim=1)                         # (B,)


# ---------------------------------------------------------------------------
# Eq. (24) — r6: Lingering Fraction Reward
# ---------------------------------------------------------------------------

def linger_reward(
    tau0: torch.Tensor,
    l_min: float,
    l_max: float,
    sigma_l: float = 0.1,
    v_linger: float = 0.5,
    k: int = 5,
    xi_l: float = 0.1,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    ℓ_soft^i = max_{t₀} ∏_{t=t₀}^{t₀+k-1} sigmoid((v̄_ℓ − v_t^i) / ξ_ℓ)
    r_linger = Band((1/N)∑_i ℓ_soft^i; ℓ_min, ℓ_max, σ_ℓ)

    Differentiable: uses max over soft product of consecutive slow-step indicators.

    Args:
        tau0:     (B, N, T, 2)
        l_min/max: desired lingering fraction band
        sigma_l:  Gaussian decay width
        v_linger: slow speed threshold (m/s)
        k:        consecutive timesteps for lingering window
        xi_l:     sigmoid sharpness
        dt:       timestep (s)
    Returns:
        (B,)
    """
    diffs  = tau0[:, :, 1:, :] - tau0[:, :, :-1, :]   # (B, N, T-1, 2)
    speeds = diffs.norm(dim=-1) / dt                    # (B, N, T-1)

    # Soft slow-step indicator: sigmoid((v_linger - v_t) / ξ)
    slow = torch.sigmoid((v_linger - speeds) / xi_l)    # (B, N, T-1)

    B, N, Tm1 = slow.shape
    if k > Tm1:
        k = Tm1

    # Compute soft product over all windows of length k
    # Use log-sum-exp trick for numerical stability
    log_slow = torch.log(slow.clamp(min=1e-8))         # (B, N, T-1)

    # Slide a window of size k and sum log-probs
    # Unfold along time: (B, N, num_windows, k)
    num_windows = Tm1 - k + 1
    windows = log_slow.unfold(dimension=2, size=k, step=1)  # (B, N, num_windows, k)
    log_prods = windows.sum(dim=-1)                          # (B, N, num_windows)

    # max over windows → best lingering window per agent
    max_log_prod = log_prods.max(dim=-1).values              # (B, N)
    ell_soft = max_log_prod.exp()                            # (B, N)  ∈ (0, 1]

    ell_bar = ell_soft.mean(dim=1)                           # (B,)
    return band_reward(ell_bar, l_min, l_max, sigma_l)


# ---------------------------------------------------------------------------
# Eq. (25) — Multiplicative Total Reward
# ---------------------------------------------------------------------------

def total_reward(
    tau0: torch.Tensor,
    # Speed
    v_min: float, v_max: float, sigma_v: float = 0.5, dt: float = 1.0,
    # Early collision
    c_min_early: float = 0.0, c_max_early: float = 0.1,
    sigma_c: float = 0.05, d_col: float = 1.0, xi_col: float = 0.1,
    # Late collision
    c_min_late: float = 0.0, c_max_late: float = 0.1,
    # Event
    event_center: torch.Tensor | None = None,
    event_direction: str = "toward",
    rho_min: float = 0.0, rho_max: float = 1.0, sigma_rho: float = 0.2, xi_e: float = 0.5,
    # Goal
    goals: torch.Tensor | None = None, d_goal: bool = False, sigma_g: float = 1.0,
    # Linger
    l_min: float = 0.0, l_max: float = 0.2, sigma_l: float = 0.05,
    v_linger: float = 0.5, k: int = 5, xi_l: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    R(τ, C, M) = ∏_{k=1}^6 r_k(τ)  — Eq. (25)

    Returns:
        R:    (B,) total reward
        subs: dict of 6 named sub-rewards, each (B,)
    """
    r1 = speed_reward(tau0, v_min, v_max, sigma_v, dt)
    r2 = collision_reward_early(tau0, c_min_early, c_max_early, sigma_c, d_col, xi_col)
    r3 = collision_reward_late(tau0, c_min_late, c_max_late, sigma_c, d_col, xi_col)

    if event_center is not None:
        r4 = event_reward(tau0, event_center, event_direction, rho_min, rho_max, sigma_rho, xi_e)
    else:
        r4 = torch.ones(tau0.shape[0], device=tau0.device)

    r5 = goal_reward(tau0, goals, d_goal, sigma_g)
    r6 = linger_reward(tau0, l_min, l_max, sigma_l, v_linger, k, xi_l, dt)

    subs = {"speed": r1, "col_early": r2, "col_late": r3, "event": r4, "goal": r5, "linger": r6}
    R = r1 * r2 * r3 * r4 * r5 * r6
    return R, subs
