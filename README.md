# CRDTraj

PyTorch implementation of **CRDTraj** — a Conditional-Reward Diffusion Transformer for context-conditioned multi-agent trajectory generation.

The model takes a top-down map, a natural-language behavioral context (e.g. *"agents should evacuate toward the exit, avoiding collisions"*), and initial agent states, then generates a full trajectory batch via guided DDPM diffusion.

---

## Repository layout

```
CRDTraj/
├── model/
│   ├── __init__.py
│   ├── diffusion.py      # DDPM schedule, q_sample, Tweedie (Eqs. 2-5)
│   ├── encoders.py       # AgentTokenizer, MapEncoder, ContextEncoder, TimestepEmbedding
│   ├── transformer.py    # TransformerBlock (self-attn + cross-attn + FFN), TransformerBackbone
│   ├── heads.py          # DenoisingHead, RewardHead
│   ├── reward.py         # band_reward + 6 sub-reward functions (Eqs. 16-25)
│   ├── controller.py     # AdaptiveController (3-layer MLP), reinforce_loss (Eqs. 29-31)
│   └── crdtraj.py        # CRDTraj main model, inference loop (Algorithm 2), stage1_loss
├── train.py              # Stage 1 + Stage 2 training, DDP, wandb/TensorBoard logging
├── verify.py             # Smoke-test: shapes, rewards in [0,1], forward + inference
└── README.md
```

---

## Environment setup

```bash
conda create -n crdtraj-py312 python=3.12 -y
conda activate crdtraj-py312
pip install torch torchvision "numpy<2" einops sentence-transformers wandb tensorboard
```

---

## Quick smoke test

```bash
conda activate crdtraj-py312
python verify.py
```

Expected output ends with:
```
=== All verification checks passed! ===
```

---

## Training

### Single GPU

```bash
conda activate crdtraj-py312
python train.py \
  --stage1_epochs 100 \
  --stage2_epochs 50 \
  --batch_size 32 \
  --d 256 \
  --L_blocks 6 \
  --n_heads 8 \
  --diffusion_T 1000 \
  --schedule cosine \
  --lr_stage1 3e-4 \
  --lr_stage2 1e-4 \
  --lambda_r 1.0 \
  --ckpt_dir checkpoints/ \
  --wandb_project crdtraj
```

### Multi-GPU (DDP via torchrun)

```bash
conda activate crdtraj-py312
torchrun --nproc_per_node=4 train.py \
  --batch_size 32 \
  --stage1_epochs 100 \
  --stage2_epochs 50 \
  --ckpt_dir checkpoints/ \
  --wandb_project crdtraj
```

Replace `--nproc_per_node=4` with the number of GPUs available.

### Resume from checkpoint

```bash
python train.py --resume checkpoints/stage1_epoch0009.pt
```

or with torchrun:

```bash
torchrun --nproc_per_node=4 train.py --resume checkpoints/stage1_epoch0099.pt
```

### Disable wandb / use TensorBoard only

```bash
python train.py --use_wandb --use_tb   # both (default)
python train.py --use_tb               # TensorBoard only (pass --use_wandb to also enable wandb)
python train.py                        # console logs only
```

TensorBoard logs are written to `checkpoints/tb/`. View with:

```bash
tensorboard --logdir checkpoints/tb
```

---

## Inference

Call the model directly from Python:

```python
import torch
from model import CRDTraj
from sentence_transformers import SentenceTransformer

# Load trained model
model = CRDTraj(
    T=20,           # trajectory length
    d=256,          # model dimension
    L_blocks=6,
    n_heads=8,
    L_ctx=16,
    sent_dim=384,   # must match encoder (all-MiniLM-L6-v2 = 384)
    diffusion_T=1000,
    schedule="cosine",
)
model.load_state_dict(torch.load("checkpoints/stage2_epoch0049.pt")["model"])
model.eval()

# Encode context text
sbert = SentenceTransformer("all-MiniLM-L6-v2")
C = sbert.encode(["agents walk toward the exit"], convert_to_tensor=True)  # (384,)
C = C.unsqueeze(0)  # (1, 384)  — batch size 1

# Prepare inputs
B, N = 1, 8
M  = torch.rand(B, 3, 224, 224)   # top-down map (ImageNet-normalised)
S0 = torch.zeros(B, N, 6)         # initial states: (x, y, vx, vy, gx, gy)

# Run guided denoising (Algorithm 2)
tau0 = model.inference(M, C, S0, Gamma=1000)   # (B, N, T, 2)
print(tau0.shape)   # torch.Size([1, 8, 20, 2])
```

`tau0[b, i, t, :]` is the `(x, y)` position of agent `i` at timestep `t` in scenario `b`.

---

## Key hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--T` | 20 | Trajectory length (timesteps) |
| `--d` | 256 | Model (token) dimension |
| `--L_blocks` | 6 | Number of transformer layers |
| `--n_heads` | 8 | Attention heads per layer |
| `--L_ctx` | 16 | Context tokens from ContextEncoder |
| `--sent_dim` | 384 | SentenceBERT output dim (`all-MiniLM-L6-v2` = 384, `all-mpnet-base-v2` = 768) |
| `--diffusion_T` | 1000 | Diffusion steps Γ |
| `--schedule` | cosine | Noise schedule: `cosine` or `linear` |
| `--lambda_r` | 1.0 | Reward loss weight λ (Eq. 32) |
| `--g_max` | 1.0 | Maximum guidance gate value |
| `--beta_ctrl` | 1e-3 | Gate regularisation coefficient β (Eq. 30) |
| `--stage1_epochs` | 100 | Epochs for joint denoising + reward training |
| `--stage2_epochs` | 50 | Epochs for controller REINFORCE training |
| `--batch_size` | 32 | Per-GPU batch size |
| `--lr_stage1` | 3e-4 | Learning rate for Stage 1 |
| `--lr_stage2` | 1e-4 | Learning rate for Stage 2 (controller only) |
| `--N_agents` | 8 | Number of agents per scenario |

---

## Training stages

**Stage 1** — joint backbone + heads training (Eq. 32):

```
L = ||ε − ε_θ||² + λ Σ_k λ_k(t) · ||r̂_k − r_k||²
```

Trains the transformer backbone, denoising head, and reward head together.
`λ_k(t)` (Eq. 33) downweights reward components that are unreliable at high noise levels.

**Stage 2** — controller training (Eqs. 30-31):

Backbone and heads are frozen. The `AdaptiveController` (~50K parameters) is trained via REINFORCE to learn per-component guidance gates `g_k(t)` that maximise final trajectory reward:

```
L_ctrl = −E[(R − R̄) · log π_ψ(g)] + β Σ_{t,k} g_k(t)²
```

---

## Logged metrics

**Stage 1** (every `--log_every` steps):

- `stage1/loss_total`
- `stage1/loss_denoise`
- `stage1/loss_reward_total`
- `stage1/loss_reward_{speed,col_early,col_late,event,goal,linger}` — per-component

**Stage 2** (every `--log_every` steps):

- `stage2/ctrl_loss`
- `stage2/reward_mean`, `stage2/reward_std`
- `stage2/reward_{speed,col_early,col_late,event,goal,linger}` — per-component means
- `stage2/gates_epoch{N}` — gate schedule heatmap (TensorBoard image)

---

## Data

`SyntheticDataset` in `train.py` is a random-data placeholder. Replace it with a real loader for:

- **ETH/UCY** pedestrian datasets
- **Stanford Drone Dataset (SDD)**
- Custom LLM-generated synthetic scenarios (Social Force Model with behavioral specifications)

Each sample must yield `(tau0, M, C, S0)`:

| Tensor | Shape | Description |
|---|---|---|
| `tau0` | `(N, T, 2)` | Clean ground-truth trajectory (x, y positions) |
| `M` | `(3, H, W)` | Top-down map image (ImageNet-normalised) |
| `C` | `(sent_dim,)` | Pre-computed SentenceBERT embedding of context string |
| `S0` | `(N, 6)` | Initial agent states: `(x₀, y₀, vx₀, vy₀, gx, gy)` |

---

## Citation

If you use this code, please cite the CRDTraj paper.
