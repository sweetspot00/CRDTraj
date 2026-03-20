"""
CRDTraj — main model combining all components.

forward(tau_t, t, M, C, S0) → (eps_pred, reward_preds, z_agents)
inference(M, C, S0, Gamma)   — Algorithm 2 guided denoising
stage1_loss(tau0, M, C, S0)  — Eq. (32) joint denoising + reward loss
"""

import torch
import torch.nn as nn

from .diffusion import DiffusionSchedule
from .encoders import AgentTokenizer, MapEncoder, ContextEncoder, TimestepEmbedding
from .transformer import TransformerBackbone
from .heads import DenoisingHead, RewardHead
from .controller import AdaptiveController


class CRDTraj(nn.Module):
    """
    Conditional-Reward Diffusion Transformer for multi-agent trajectory generation.

    Combines:
      AgentTokenizer    — Eq. (6)
      MapEncoder        — Eq. (7)
      ContextEncoder    — Eq. (8)
      TimestepEmbedding — Eq. (9)
      TransformerBackbone — Eqs. (11-13)
      DenoisingHead     — Eq. (14)
      RewardHead        — Eq. (15)
      AdaptiveController — Eq. (29)
      DiffusionSchedule — Eqs. (2-5)

    Args:
        T:          trajectory length (number of timesteps)
        d:          model dimension
        L_blocks:   number of transformer layers
        n_heads:    attention heads
        L_ctx:      number of context tokens produced by ContextEncoder
        sent_dim:   SentenceBERT embedding dimension
        P_map:      expected number of map patch tokens (for reference; dynamic)
        diffusion_T: number of diffusion steps Γ
        schedule:   'cosine' or 'linear'
        g_max:      max gate value for AdaptiveController
        lambda_r:   overall reward loss weight (λ in Eq. 32)
        map_frozen: freeze ResNet-18 weights
        sbert_model: SentenceBERT model name
    """

    def __init__(
        self,
        T: int = 20,
        d: int = 256,
        L_blocks: int = 6,
        n_heads: int = 8,
        L_ctx: int = 16,
        sent_dim: int = 384,
        diffusion_T: int = 1000,
        schedule: str = "cosine",
        g_max: float = 1.0,
        lambda_r: float = 1.0,
        map_frozen: bool = False,
        sbert_model: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()

        self.T_traj = T
        self.d = d
        self.lambda_r = lambda_r

        # Diffusion schedule
        self.schedule = DiffusionSchedule(T=diffusion_T, schedule=schedule)

        # Encoders
        self.agent_tokenizer = AgentTokenizer(T=T, d=d)
        self.map_encoder = MapEncoder(d=d, frozen=map_frozen)
        self.ctx_encoder = ContextEncoder(d=d, L=L_ctx, sent_dim=sent_dim, sbert_model_name=sbert_model)
        self.time_emb = TimestepEmbedding(d=d)

        # Backbone
        self.backbone = TransformerBackbone(d=d, L=L_blocks, n_heads=n_heads)

        # Heads
        self.denoise_head = DenoisingHead(d=d, T=T)
        self.reward_head = RewardHead(d=d)

        # Controller (w_ψ, trained in Stage 2)
        self.controller = AdaptiveController(d=d, g_max=g_max)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        tau_t: torch.Tensor,
        t: torch.Tensor,
        M: torch.Tensor,
        C: torch.Tensor,
        S0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single diffusion forward pass.

        Args:
            tau_t: (B, N, T, 2)  noisy trajectories at diffusion step t
            t:     (B,)          diffusion timestep indices
            M:     (B, 3, H, W)  map images
            C:     (B, sent_dim) pre-computed context embeddings
            S0:    (B, N, 6)     initial agent states (x0,y0,vx0,vy0,gx,gy)

        Returns:
            eps_pred:     (B, N, T, 2)  predicted noise ε_θ
            reward_preds: (B, 6)        predicted sub-rewards r̂_k ∈ [0,1]
            z_agents:     (B, N, d)     agent representations (for controller)
        """
        B, N, Traj, _ = tau_t.shape

        # --- Encode inputs ---
        h_agent = self.agent_tokenizer(tau_t, S0)    # (B, N, d)  Eq. (6)
        h_map   = self.map_encoder(M)                # (B, P, d)  Eq. (7)
        h_ctx   = self.ctx_encoder(C)                # (B, L, d)  Eq. (8)
        h_t     = self.time_emb(t)                   # (B, d)     Eq. (9)

        # Add timestep embedding to all agent tokens (global signal, Eq. 10)
        h_agent = h_agent + h_t.unsqueeze(1)         # (B, N, d)

        # Concatenate map + context tokens for cross-attention keys/values
        H_ctx = torch.cat([h_map, h_ctx], dim=1)     # (B, P+L, d)

        # --- Transformer backbone --- Eqs. (11-13)
        z_agents = self.backbone(h_agent, H_ctx)     # (B, N, d)

        # --- Denoising head --- Eq. (14)
        eps_pred = self.denoise_head(z_agents)        # (B, N, T, 2)

        # --- Tweedie estimate τ̂₀ --- Eq. (5)
        tau0_hat = self.schedule.predict_x0(tau_t, t, eps_pred)  # (B, N, T, 2)

        # --- Reward head --- Eq. (15)
        reward_preds = self.reward_head(z_agents, tau0_hat)      # (B, 6)

        return eps_pred, reward_preds, z_agents

    # ------------------------------------------------------------------
    # Stage 1 loss — Eq. (32)
    # ------------------------------------------------------------------

    def stage1_loss(
        self,
        tau0: torch.Tensor,
        M: torch.Tensor,
        C: torch.Tensor,
        S0: torch.Tensor,
        gt_rewards: dict[str, torch.Tensor],
        kappa: torch.Tensor | None = None,
        xi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        L = ‖ε − ε_θ‖² + λ ∑_k λ_k(t) · ‖r̂_k − r_k‖²  — Eq. (32)

        λ_k(t) = sigmoid((SNR(t) − κ_k) / ξ_k)           — Eq. (33)

        Args:
            tau0:       (B, N, T, 2) clean trajectories
            M, C, S0:   map, context, initial states
            gt_rewards: dict of 6 ground-truth sub-rewards, each (B,)
                        keys: 'speed','col_early','col_late','event','goal','linger'
            kappa:      (6,) SNR thresholds per component (Eq. 33); default zeros
            xi:         (6,) sharpness per component;                default ones

        Returns:
            loss:     scalar
            log_dict: per-component losses for logging
        """
        B = tau0.shape[0]
        device = tau0.device

        # Sample random diffusion timesteps
        t = torch.randint(0, self.schedule.T, (B,), device=device)

        # Forward noising  τ_t = √ᾱ_t · τ_0 + √(1-ᾱ_t) · ε
        eps = torch.randn_like(tau0)
        tau_t = self.schedule.q_sample(tau0, t, eps)

        # Model forward
        eps_pred, reward_preds, _ = self.forward(tau_t, t, M, C, S0)

        # Denoising loss
        loss_denoise = ((eps - eps_pred) ** 2).mean()

        # Noise-level-dependent weights λ_k(t) — Eq. (33)
        snr_t = self.schedule.snr(t)  # (B,)
        if kappa is None:
            kappa = torch.zeros(6, device=device)
        if xi is None:
            xi = torch.ones(6, device=device)

        # λ_k(t): (B, 6)
        lambda_kt = torch.sigmoid(
            (snr_t.unsqueeze(1) - kappa.unsqueeze(0)) / xi.unsqueeze(0)
        )

        # Reward prediction loss for each of 6 components
        gt_tensor = torch.stack([
            gt_rewards["speed"],
            gt_rewards["col_early"],
            gt_rewards["col_late"],
            gt_rewards["event"],
            gt_rewards["goal"],
            gt_rewards["linger"],
        ], dim=1)  # (B, 6)

        reward_sq_err = (reward_preds - gt_tensor) ** 2   # (B, 6)
        weighted_reward_loss = (lambda_kt * reward_sq_err).mean(dim=0)  # (6,)
        loss_reward = weighted_reward_loss.sum()

        loss_total = loss_denoise + self.lambda_r * loss_reward

        keys = ["speed", "col_early", "col_late", "event", "goal", "linger"]
        log_dict = {"loss_denoise": loss_denoise, "loss_reward_total": loss_reward}
        for i, k in enumerate(keys):
            log_dict[f"loss_reward_{k}"] = weighted_reward_loss[i]

        return loss_total, log_dict

    # ------------------------------------------------------------------
    # Algorithm 2 — Inference
    # ------------------------------------------------------------------

    def inference(
        self,
        M: torch.Tensor,
        C: torch.Tensor,
        S0: torch.Tensor,
        Gamma: int | None = None,
        reward_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """
        Guided denoising loop — Algorithm 2.

        1. Encode context to get e_ctx
        2. Sample τ_Γ ~ N(0, I)
        3. For t = Γ, ..., 1:
           a. Forward pass → ε_θ, r̂_k
           b. Tweedie → τ̂₀
           c. Compute guidance gradients ∇_{τ_t} r̂_k
           d. Compute gates g_k via controller
           e. p_sample with guidance
        4. Return τ₀

        Args:
            M:            (B, 3, H, W) map
            C:            (B, sent_dim) context embedding
            S0:           (B, N, 6) initial states
            Gamma:        number of diffusion steps (defaults to schedule.T)
            reward_kwargs: extra kwargs forwarded to reward head pooling (unused here)
        Returns:
            tau0: (B, N, T, 2)
        """
        B, N, _ = S0.shape
        T_traj = self.T_traj
        device = M.device
        Gamma = Gamma or self.schedule.T

        # Pre-encode context — no gradients needed here
        with torch.no_grad():
            h_ctx = self.ctx_encoder(C)                   # (B, L, d)
            h_map = self.map_encoder(M)                   # (B, P, d)
            H_ctx = torch.cat([h_map, h_ctx], dim=1)      # (B, P+L, d)
            e_ctx = h_ctx.mean(dim=1)                     # (B, d)

        # Initial noise
        tau_t = torch.randn(B, N, T_traj, 2, device=device)

        for step in reversed(range(Gamma)):
            t = torch.full((B,), step, dtype=torch.long, device=device)

            # Enable autograd w.r.t. tau_t for guidance gradient computation
            tau_t_g = tau_t.detach().requires_grad_(True)

            with torch.enable_grad():
                # Agent tokenization + timestep emb
                h_agent = self.agent_tokenizer(tau_t_g, S0) + self.time_emb(t).unsqueeze(1)

                # Backbone
                z_agents = self.backbone(h_agent, H_ctx)

                # Denoising head
                eps_pred = self.denoise_head(z_agents)

                # Tweedie
                tau0_hat = self.schedule.predict_x0(tau_t_g, t, eps_pred)

                # Reward head
                reward_preds = self.reward_head(z_agents, tau0_hat)  # (B, 6)

                # Compute per-component reward gradients w.r.t. tau_t — Eq. (27)
                R_hat = reward_preds.prod(dim=1)                     # (B,)
                grad_norms = torch.zeros(B, 6, device=device)
                grads_per_k = []
                for k in range(6):
                    g = torch.autograd.grad(
                        reward_preds[:, k].sum(),
                        tau_t_g,
                        retain_graph=(k < 5),
                        create_graph=False,
                    )[0]  # (B, N, T, 2)
                    # Eq. (27): scale by 1/√ᾱ_t
                    sqrt_ab = self.schedule.sqrt_alphas_bar[t]  # (B,)
                    while sqrt_ab.ndim < g.ndim:
                        sqrt_ab = sqrt_ab.unsqueeze(-1)
                    g = g / sqrt_ab
                    grad_norms[:, k] = g.reshape(B, -1).norm(dim=1)
                    grads_per_k.append(g.detach())

                # Adaptive gates — Eq. (29)
                snr_t = self.schedule.snr(t)  # (B,)
                gates = self.controller(snr_t, e_ctx, R_hat.detach(), grad_norms)  # (B, 6)

            # Combined guidance — Eq. (28): ∑_k g_k · ∇_{τ_t} r̂_k  (no grad needed)
            guidance = sum(
                gates[:, k].detach().reshape(B, 1, 1, 1) * grads_per_k[k]
                for k in range(6)
            )

            # Reverse step
            tau_t = self.schedule.p_sample(
                tau_t_g.detach(), t,
                eps_pred.detach(),
                guidance=guidance,
            ).detach()

        return tau_t  # (B, N, T, 2)
