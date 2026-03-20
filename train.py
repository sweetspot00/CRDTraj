"""
CRDTraj Training Script.

Stage 1: Joint backbone + heads (denoising + reward prediction) — Eq. (32)
Stage 2: AdaptiveController via REINFORCE — Eqs. (30-31)

Launch with torchrun:
  torchrun --nproc_per_node=<N_GPUS> train.py [--config config.yaml]

Logging:
  wandb:       per-step loss curves, per-component reward bars, gate heatmaps
  TensorBoard: same metrics via SummaryWriter (alternative)
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Optional imports — gracefully degrade if unavailable
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from model import CRDTraj
from model.reward import total_reward
from model.controller import reinforce_loss


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # Architecture
    T=20,
    d=256,
    L_blocks=6,
    n_heads=8,
    L_ctx=16,
    sent_dim=384,
    diffusion_T=1000,
    schedule="cosine",
    g_max=1.0,
    lambda_r=1.0,
    # Training
    stage1_epochs=100,
    stage2_epochs=50,
    batch_size=32,
    lr_stage1=3e-4,
    lr_stage2=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    log_every=50,
    save_every=1,
    # Controller
    beta_ctrl=1e-3,
    rollout_batch=16,
    # Data
    data_dir="data/",
    ckpt_dir="checkpoints/",
    resume=None,
    # Misc
    seed=42,
    num_workers=4,
    wandb_project="crdtraj",
    wandb_run=None,
    use_wandb=True,
    use_tb=True,
    N_agents=8,
)


# ---------------------------------------------------------------------------
# Minimal synthetic dataset (placeholder — replace with real data loader)
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Generates random trajectories, maps, and context embeddings on-the-fly.
    Replace this with a real dataset loader for ETH/UCY or SDD.
    """

    def __init__(self, size: int, T: int, N: int, d_map: int = 224, sent_dim: int = 384):
        self.size = size
        self.T = T
        self.N = N
        self.d_map = d_map
        self.sent_dim = sent_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tau0  = torch.randn(self.N, self.T, 2)          # (N, T, 2) clean traj
        M     = torch.rand(3, self.d_map, self.d_map)   # (3, H, W) map image
        C     = torch.randn(self.sent_dim)               # (sent_dim,) context emb
        S0    = torch.randn(self.N, 6)                   # (N, 6) initial states
        return tau0, M, C, S0


def build_gt_rewards(tau0: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute ground-truth rewards on clean trajectories for Stage 1 training."""
    _, subs = total_reward(
        tau0,
        v_min=0.5, v_max=2.0, sigma_v=0.5, dt=1.0,
        c_min_early=0.0, c_max_early=0.1, sigma_c=0.05, d_col=1.0, xi_col=0.1,
        c_min_late=0.0, c_max_late=0.1,
        d_goal=False,
        l_min=0.0, l_max=0.2, sigma_l=0.05, v_linger=0.5, k=3, xi_l=0.1,
    )
    return subs


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class Logger:
    """Thin wrapper around wandb + TensorBoard."""

    def __init__(self, cfg: dict, use_wandb: bool, use_tb: bool):
        self.wb = None
        self.tb = None

        if use_wandb and WANDB_AVAILABLE and is_main():
            self.wb = wandb.init(
                project=cfg["wandb_project"],
                name=cfg.get("wandb_run"),
                config=cfg,
            )

        if use_tb and TB_AVAILABLE and is_main():
            tb_dir = Path(cfg["ckpt_dir"]) / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb = SummaryWriter(str(tb_dir))

    def log(self, metrics: dict, step: int):
        if not is_main():
            return
        if self.wb is not None:
            wandb.log(metrics, step=step)
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.tb.add_scalar(k, v, step)

    def log_gate_heatmap(self, gates: torch.Tensor, step: int, tag: str = "gates/heatmap"):
        """Log gate schedule heatmap (gates shape: (T_steps, 6))."""
        if not is_main():
            return
        if self.tb is not None:
            # gates: (T_steps, 6) → add image
            img = gates.unsqueeze(0).unsqueeze(0)  # (1, 1, T, 6)
            self.tb.add_image(tag, img, step, dataformats="NCHW")

    def finish(self):
        if self.wb is not None:
            wandb.finish()
        if self.tb is not None:
            self.tb.close()


# ---------------------------------------------------------------------------
# Stage 1: Joint training
# ---------------------------------------------------------------------------

def train_stage1(model, loader, optimizer, scheduler, logger, cfg, start_epoch=0):
    device = next(model.parameters()).device
    step = start_epoch * len(loader)

    for epoch in range(start_epoch, cfg["stage1_epochs"]):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        model.train()
        for batch in loader:
            tau0, M, C, S0 = [x.to(device) for x in batch]

            # Ground-truth rewards on clean trajectories
            with torch.no_grad():
                gt_rewards = build_gt_rewards(tau0)

            optimizer.zero_grad()
            loss, log_dict = model.module.stage1_loss(tau0, M, C, S0, gt_rewards) if isinstance(model, DDP) \
                             else model.stage1_loss(tau0, M, C, S0, gt_rewards)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            if step % cfg["log_every"] == 0:
                metrics = {f"stage1/{k}": v for k, v in log_dict.items()}
                metrics["stage1/loss_total"] = loss.item()
                metrics["stage1/lr"] = optimizer.param_groups[0]["lr"]
                logger.log(metrics, step)

                if is_main():
                    print(
                        f"[Stage1] epoch={epoch} step={step} "
                        f"loss={loss.item():.4f} "
                        f"l_denoise={log_dict['loss_denoise'].item():.4f} "
                        f"l_reward={log_dict['loss_reward_total'].item():.4f}"
                    )
            step += 1

        if scheduler is not None:
            scheduler.step()

        if is_main() and (epoch + 1) % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, step, cfg, stage=1)

    return step


# ---------------------------------------------------------------------------
# Stage 2: Controller REINFORCE training
# ---------------------------------------------------------------------------

def train_stage2(model, loader, optimizer, logger, cfg, start_epoch=0):
    """
    Freeze backbone + heads; train controller w_ψ via REINFORCE — Eq. (30-31).
    """
    device = next(model.parameters()).device
    raw_model = model.module if isinstance(model, DDP) else model

    # Freeze backbone + heads; only controller parameters get gradients
    for name, p in raw_model.named_parameters():
        p.requires_grad = "controller" in name

    step = start_epoch * len(loader)
    baseline = 0.0
    alpha_baseline = 0.05  # EMA decay for running baseline

    for epoch in range(start_epoch, cfg["stage2_epochs"]):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        raw_model.eval()       # backbone in eval mode
        raw_model.controller.train()

        gate_history = []      # for heatmap logging

        for batch in loader:
            _, M, C, S0 = [x.to(device) for x in batch]
            B = M.shape[0]

            optimizer.zero_grad()

            # Run full inference rollout to collect gates and log-probs
            all_log_probs = torch.zeros(B, device=device)
            all_gates = []

            tau_t = torch.randn(B, raw_model.N_agents if hasattr(raw_model, "N_agents") else S0.shape[1],
                                raw_model.T_traj, 2, device=device)

            h_ctx = raw_model.ctx_encoder(C)
            h_map = raw_model.map_encoder(M)
            H_ctx = torch.cat([h_map, h_ctx], dim=1)
            e_ctx = h_ctx.mean(dim=1)

            Gamma = raw_model.schedule.T

            for s in reversed(range(Gamma)):
                t_vec = torch.full((B,), s, dtype=torch.long, device=device)

                tau_t_g = tau_t.detach().requires_grad_(True)
                h_agent = raw_model.agent_tokenizer(tau_t_g, S0) + raw_model.time_emb(t_vec).unsqueeze(1)
                z_agents = raw_model.backbone(h_agent, H_ctx)
                eps_pred = raw_model.denoise_head(z_agents)
                tau0_hat = raw_model.schedule.predict_x0(tau_t_g, t_vec, eps_pred)
                reward_preds = raw_model.reward_head(z_agents, tau0_hat)
                R_hat = reward_preds.prod(dim=1)

                # Grad norms
                grad_norms = torch.zeros(B, 6, device=device)
                grads_per_k = []
                for k in range(6):
                    g = torch.autograd.grad(
                        reward_preds[:, k].sum(), tau_t_g,
                        retain_graph=(k < 5), create_graph=False,
                    )[0]
                    sqrt_ab = raw_model.schedule.sqrt_alphas_bar[t_vec]
                    while sqrt_ab.ndim < g.ndim:
                        sqrt_ab = sqrt_ab.unsqueeze(-1)
                    g = g / sqrt_ab
                    grad_norms[:, k] = g.reshape(B, -1).norm(dim=1)
                    grads_per_k.append(g)

                snr_t = raw_model.schedule.snr(t_vec)
                gates = raw_model.controller(snr_t, e_ctx, R_hat, grad_norms)  # (B, 6)
                log_p = raw_model.controller.log_prob(gates.detach(), snr_t, e_ctx, R_hat, grad_norms)
                all_log_probs = all_log_probs + log_p
                all_gates.append(gates.detach().mean(dim=0))  # (6,) for logging

                guidance = sum(
                    gates[:, k].reshape(B, 1, 1, 1) * grads_per_k[k]
                    for k in range(6)
                )
                with torch.no_grad():
                    tau_t = raw_model.schedule.p_sample(
                        tau_t_g.detach(), t_vec,
                        eps_pred.detach(),
                        guidance=guidance.detach(),
                    )

            # Final trajectory reward
            with torch.no_grad():
                R_final, subs = total_reward(
                    tau_t,
                    v_min=0.5, v_max=2.0, sigma_v=0.5, dt=1.0,
                    c_min_early=0.0, c_max_early=0.1, sigma_c=0.05, d_col=1.0, xi_col=0.1,
                    c_min_late=0.0, c_max_late=0.1,
                    d_goal=False,
                    l_min=0.0, l_max=0.2, sigma_l=0.05, v_linger=0.5, k=3, xi_l=0.1,
                )

            # Update baseline (EMA)
            baseline = (1 - alpha_baseline) * baseline + alpha_baseline * R_final.mean().item()

            # REINFORCE loss — Eq. (30-31)
            gates_tensor = torch.stack(all_gates, dim=0)  # (Gamma, 6)
            ctrl_loss = reinforce_loss(
                R_final, all_log_probs,
                gates_tensor.unsqueeze(0).expand(B, -1, -1),  # (B, Gamma, 6)
                baseline=baseline,
                beta=cfg["beta_ctrl"],
            )
            ctrl_loss.backward()
            nn.utils.clip_grad_norm_(raw_model.controller.parameters(), cfg["grad_clip"])
            optimizer.step()

            if step % cfg["log_every"] == 0:
                metrics = {
                    "stage2/ctrl_loss": ctrl_loss.item(),
                    "stage2/reward_mean": R_final.mean().item(),
                    "stage2/reward_std": R_final.std().item(),
                    "stage2/baseline": baseline,
                }
                for k, v in subs.items():
                    metrics[f"stage2/reward_{k}"] = v.mean().item()
                logger.log(metrics, step)

                # Gate schedule heatmap (mean over batch)
                gate_history.append(gates_tensor.cpu())

                if is_main():
                    print(
                        f"[Stage2] epoch={epoch} step={step} "
                        f"ctrl_loss={ctrl_loss.item():.4f} "
                        f"R_mean={R_final.mean().item():.4f}"
                    )
            step += 1

        # Log gate heatmap for this epoch
        if gate_history and is_main():
            gate_stack = torch.stack(gate_history, dim=0)  # (#logs, Gamma, 6)
            logger.log_gate_heatmap(gate_stack[-1], step, tag=f"stage2/gates_epoch{epoch}")

        if is_main() and (epoch + 1) % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, step, cfg, stage=2)

    # Unfreeze all parameters for any downstream use
    for p in raw_model.parameters():
        p.requires_grad = True

    return step


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, step, cfg, stage: int):
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    path = ckpt_dir / f"stage{stage}_epoch{epoch:04d}.pt"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "stage": stage,
    }, str(path))
    print(f"[ckpt] saved {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("stage", 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true" if not v else "store_false")
        else:
            p.add_argument(f"--{k}", default=v, type=t)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)

    # DDP setup
    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    torch.manual_seed(cfg["seed"] + local_rank)

    # Build model
    model = CRDTraj(
        T=cfg["T"],
        d=cfg["d"],
        L_blocks=cfg["L_blocks"],
        n_heads=cfg["n_heads"],
        L_ctx=cfg["L_ctx"],
        sent_dim=cfg["sent_dim"],
        diffusion_T=cfg["diffusion_T"],
        schedule=cfg["schedule"],
        g_max=cfg["g_max"],
        lambda_r=cfg["lambda_r"],
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Dataset + DataLoader
    dataset = SyntheticDataset(
        size=10000,
        T=cfg["T"],
        N=cfg["N_agents"],
        sent_dim=cfg["sent_dim"],
    )
    sampler = DistributedSampler(dataset) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    logger = Logger(cfg, cfg["use_wandb"], cfg["use_tb"])

    # ---- Stage 1 ----
    raw_model = model.module if use_ddp else model
    opt1 = torch.optim.AdamW(
        [p for n, p in raw_model.named_parameters() if "controller" not in n],
        lr=cfg["lr_stage1"], weight_decay=cfg["weight_decay"],
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg["stage1_epochs"])

    start_epoch = 0
    if cfg["resume"]:
        start_epoch, _, _ = load_checkpoint(cfg["resume"], model, opt1)

    if is_main():
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        print("=== Stage 1: Joint denoising + reward training ===")

    train_stage1(model, loader, opt1, sched1, logger, cfg, start_epoch)

    # ---- Stage 2 ----
    opt2 = torch.optim.AdamW(
        raw_model.controller.parameters(),
        lr=cfg["lr_stage2"], weight_decay=cfg["weight_decay"],
    )

    if is_main():
        print("=== Stage 2: Controller REINFORCE training ===")

    train_stage2(model, loader, opt2, logger, cfg)

    logger.finish()
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
