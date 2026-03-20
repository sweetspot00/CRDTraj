"""
Verification script: instantiate CRDTraj and run forward pass + reward checks.

Run with:  python verify.py
"""

import torch
from model import CRDTraj
from model.reward import band_reward, speed_reward, collision_reward_early, total_reward
from model.diffusion import DiffusionSchedule


def check_rewards():
    print("--- Reward function checks ---")
    B, N, T = 2, 4, 20

    # band_reward: should be 1.0 inside band
    x = torch.tensor([0.5, 1.5, 2.5])
    r = band_reward(x, a=1.0, b=2.0, sigma=0.3)
    print(f"band_reward([0.5, 1.5, 2.5], a=1, b=2): {r.tolist()}")
    assert r[1].item() == 1.0, "band_reward should be 1.0 inside [a,b]"
    assert (r >= 0).all() and (r <= 1).all(), "band_reward out of [0,1]"

    tau0 = torch.randn(B, N, T, 2)

    # Speed reward
    r_s = speed_reward(tau0, v_min=0.0, v_max=10.0, sigma_v=1.0)
    assert r_s.shape == (B,), f"speed_reward shape mismatch: {r_s.shape}"
    assert (r_s >= 0).all() and (r_s <= 1).all(), "speed_reward out of [0,1]"
    print(f"speed_reward: {r_s.tolist()}")

    # Total reward
    R, subs = total_reward(
        tau0, v_min=0.0, v_max=10.0, sigma_v=1.0,
        c_min_early=0.0, c_max_early=1.0, sigma_c=0.1, d_col=0.1, xi_col=0.05,
        c_min_late=0.0, c_max_late=1.0,
        d_goal=False,
        l_min=0.0, l_max=1.0, sigma_l=0.1,
    )
    assert R.shape == (B,), f"total_reward shape: {R.shape}"
    assert (R >= 0).all() and (R <= 1).all(), "total_reward out of [0,1]"
    print(f"total_reward: {R.tolist()}")
    for k, v in subs.items():
        assert (v >= 0).all() and (v <= 1).all(), f"{k} out of [0,1]"
    print("All reward checks passed.")


def check_diffusion():
    print("\n--- Diffusion schedule checks ---")
    sched = DiffusionSchedule(T=100, schedule="cosine")
    B = 4
    tau0 = torch.randn(B, 3, 10, 2)
    t    = torch.randint(0, 100, (B,))
    eps  = torch.randn_like(tau0)

    tau_t = sched.q_sample(tau0, t, eps)
    assert tau_t.shape == tau0.shape

    tau0_hat = sched.predict_x0(tau_t, t, eps)
    assert tau0_hat.shape == tau0.shape
    print(f"q_sample shape: {tau_t.shape}")
    print(f"predict_x0 reconstruction error: {(tau0 - tau0_hat).abs().max().item():.2e}  (should be ~0)")

    snr = sched.snr(t)
    assert snr.shape == (B,)
    print(f"SNR values: {snr.tolist()}")
    print("Diffusion checks passed.")


def check_model():
    print("\n--- Model forward pass ---")
    B, N, T = 2, 4, 20
    d = 128   # smaller for fast verification
    sent_dim = 64

    model = CRDTraj(
        T=T, d=d, L_blocks=2, n_heads=4, L_ctx=8,
        sent_dim=sent_dim, diffusion_T=100, schedule="cosine",
    )
    model.eval()

    tau_t = torch.randn(B, N, T, 2)
    t     = torch.randint(0, 100, (B,))
    M     = torch.rand(B, 3, 224, 224)
    C     = torch.randn(B, sent_dim)
    S0    = torch.randn(B, N, 6)

    with torch.no_grad():
        eps_pred, reward_preds, z_agents = model(tau_t, t, M, C, S0)

    print(f"eps_pred shape:     {eps_pred.shape}   (expected {(B, N, T, 2)})")
    print(f"reward_preds shape: {reward_preds.shape}  (expected {(B, 6)})")
    print(f"z_agents shape:     {z_agents.shape}   (expected {(B, N, d)})")
    assert eps_pred.shape == (B, N, T, 2)
    assert reward_preds.shape == (B, 6)
    assert z_agents.shape == (B, N, d)
    assert (reward_preds >= 0).all() and (reward_preds <= 1).all(), "reward_preds out of [0,1]"
    print("Forward pass checks passed.")


def check_stage1_loss():
    print("\n--- Stage 1 loss ---")
    B, N, T = 2, 4, 20
    d = 128
    sent_dim = 64

    model = CRDTraj(
        T=T, d=d, L_blocks=2, n_heads=4, L_ctx=8,
        sent_dim=sent_dim, diffusion_T=100,
    )
    tau0 = torch.randn(B, N, T, 2)
    M    = torch.rand(B, 3, 224, 224)
    C    = torch.randn(B, sent_dim)
    S0   = torch.randn(B, N, 6)

    from model.reward import total_reward
    _, subs = total_reward(
        tau0, v_min=0.0, v_max=10.0, sigma_v=1.0,
        c_min_early=0.0, c_max_early=1.0, sigma_c=0.1, d_col=0.1, xi_col=0.05,
        c_min_late=0.0, c_max_late=1.0,
        d_goal=False,
        l_min=0.0, l_max=1.0, sigma_l=0.1,
    )

    loss, log_dict = model.stage1_loss(tau0, M, C, S0, subs)
    print(f"Stage 1 total loss: {loss.item():.4f}")
    for k, v in log_dict.items():
        print(f"  {k}: {v.item():.4f}")
    assert loss.isfinite(), "loss is non-finite"
    print("Stage 1 loss check passed.")


def check_inference():
    print("\n--- Inference (short rollout) ---")
    B, N, T = 1, 3, 10
    d = 64
    sent_dim = 32
    Gamma = 5   # very short for speed

    model = CRDTraj(
        T=T, d=d, L_blocks=2, n_heads=4, L_ctx=4,
        sent_dim=sent_dim, diffusion_T=Gamma,
    )
    model.eval()

    M  = torch.rand(B, 3, 224, 224)
    C  = torch.randn(B, sent_dim)
    S0 = torch.randn(B, N, 6)

    tau0 = model.inference(M, C, S0, Gamma=Gamma)
    print(f"Inference output shape: {tau0.shape}  (expected {(B, N, T, 2)})")
    assert tau0.shape == (B, N, T, 2), f"shape mismatch: {tau0.shape}"
    print("Inference check passed.")


if __name__ == "__main__":
    check_rewards()
    check_diffusion()
    check_model()
    check_stage1_loss()
    check_inference()
    print("\n=== All verification checks passed! ===")
