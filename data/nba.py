# obs_len / pred_len split is only for benchmark eval — training uses the full seq_len window
"""
NBA SportVU Player-Tracking Dataset.

Data source
-----------
SportVU tracking data from the 2015–16 NBA season.
Each game file is a JSON with player positions sampled at 25 Hz.

Download
--------
The processed version used by trajectory-prediction papers is available at:
    https://github.com/linouk23/NBA-Player-Movements
    (raw JSON: data/2016.NBA.Raw.SportVU.Game.Logs/)

Or the preprocessed numpy version from GroupNet / EvolveGraph:
    Each file: nba_{split}.npy  shape (N_sequences, 11, 50, 2)
    11 agents = 5 home + 5 away + 1 ball
    50 frames at 25 Hz → 2 seconds
    Coordinates in feet: [0, 94] × [0, 50]

Map
---
There is no pre-recorded aerial image of an NBA court, so we render
the court geometry programmatically. The court map shows:
  - 1.0 = in-bounds playing area (walkable)
  - 0.0 = out-of-bounds / walls

Court dimensions: 94 ft × 50 ft (28.65 m × 15.24 m)
Coordinate system: (0,0) = corner; (94,50) = opposite corner (in feet),
or converted to metres if use_metres=True (default).

Context
-------
A fixed description per game is used as the SBERT context.
"""

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Court geometry constants
# ---------------------------------------------------------------------------

COURT_FT_W  = 94.0   # feet
COURT_FT_H  = 50.0   # feet
FT_TO_M     = 0.3048
COURT_M_W   = COURT_FT_W * FT_TO_M   # 28.65 m
COURT_M_H   = COURT_FT_H * FT_TO_M   # 15.24 m

MAP_SIZE    = 224
DT_25HZ     = 1.0 / 25.0    # 25 Hz raw
TARGET_DT   = 0.4            # resample to 0.4 s (every 10 frames)

NBA_DESCRIPTION = "Basketball players moving dynamically on an NBA court during a game"


# ---------------------------------------------------------------------------
# Court map
# ---------------------------------------------------------------------------

def build_court_map(size: int = MAP_SIZE) -> torch.Tensor:
    """
    Render a simple NBA court as a (3, size, size) float32 tensor.

    The entire in-bounds area is walkable (1.0). The map has no internal
    obstacles; trajectory-prediction models learn to stay in bounds from data.

    Returns (3, size, size) in [0, 1].
    """
    # Full court = all walkable
    grid = torch.ones(size, size)
    return grid.unsqueeze(0).expand(3, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def feet_to_metres(traj: np.ndarray) -> np.ndarray:
    """Convert trajectories from feet to metres."""
    return traj * FT_TO_M


# ---------------------------------------------------------------------------
# Load preprocessed numpy format  (N_seq, 11, 50, 2)
# ---------------------------------------------------------------------------

def load_nba_npy(npy_path: Path, use_metres: bool = True) -> np.ndarray:
    """
    Load a preprocessed NBA numpy file.

    Expected shape: (N_seq, N_agents, T_frames, 2)
    Coordinates in feet; converted to metres if use_metres=True.

    Returns (N_seq, N_agents, T_frames, 2) float32.
    """
    data = np.load(npy_path).astype(np.float32)
    if data.ndim == 3:
        # Some versions store (N_seq * N_agents, T, 2) — reshape
        raise ValueError(
            f"Unexpected shape {data.shape}. Expected (N_seq, N_agents, T, 2)."
        )
    if use_metres:
        data = data * FT_TO_M
    return data


def _extract_windows_npy(
    sequences: np.ndarray,   # (N_seq, N_agents, T_raw, 2)
    seq_len: int,
    target_dt: float,
    raw_dt: float,
    max_agents: int | None,
) -> list[np.ndarray]:
    """
    Subsample raw 25 Hz sequences to target_dt and slice to seq_len frames.
    Each raw sequence is 50 frames (2 s at 25 Hz).
    """
    stride = max(1, round(target_dt / raw_dt))  # e.g. 10 for 0.4 s / 0.04 s
    windows = []

    for seq in sequences:                        # (N_agents, T_raw, 2)
        sub = seq[:, ::stride, :]               # (N_agents, T_sub, 2)
        T_sub = sub.shape[1]
        if T_sub < seq_len:
            continue
        for start in range(0, T_sub - seq_len + 1, seq_len):
            w = sub[:, start: start + seq_len, :]  # (N_agents, seq_len, 2)
            if max_agents is not None:
                w = w[:max_agents]
            windows.append(w.astype(np.float32))

    return windows


def _embed(text: str, sent_dim: int, sbert, cache_dir: Path) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key   = hashlib.md5(text.encode()).hexdigest()[:16]
    fpath = cache_dir / f"{key}.pt"
    if fpath.exists():
        return torch.load(str(fpath), map_location="cpu", weights_only=True)
    if sbert is not None:
        with torch.no_grad():
            emb = sbert.encode(text, convert_to_tensor=True).cpu().float()
    else:
        emb = torch.zeros(sent_dim)
    torch.save(emb, str(fpath))
    return emb


def _try_load_sbert(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        print("[NBA] Warning: sentence-transformers not available; using zero context.")
        return None


# ---------------------------------------------------------------------------
# NBADataset
# ---------------------------------------------------------------------------

class NBADataset(Dataset):
    """
    NBA SportVU player-tracking dataset.

    Expects preprocessed .npy files in nba_root/:
        train.npy   shape (N, 11, 50, 2)  — 11 agents, 50 frames @ 25 Hz, feet
        test.npy    shape (N, 11, 50, 2)

    Each sample returns (tau0, M, C, S0):
      tau0 : (N_agents, T, 2)  trajectory in metres    float32
      M    : (3, 224, 224)     court map (all walkable) float32
      C    : (sent_dim,)       SBERT game description   float32
      S0   : (N_agents, 6)     [x0,y0,vx0,vy0,gx,gy]  float32

    Parameters
    ----------
    nba_root    : directory containing train.npy / test.npy
    split       : 'train' | 'val' | 'test'
    seq_len     : total trajectory window length in frames (at target_dt)
    target_dt   : resample raw 25 Hz data to this time step (seconds)
    max_agents  : truncate agents; default None keeps all 11
    use_metres  : convert feet to metres (default True)
    val_frac    : fraction of train.npy held out for validation
    """

    def __init__(
        self,
        nba_root: str,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 15,    # 6 s total at 0.4 s/frame
        target_dt: float = TARGET_DT,
        min_agents: int = 5,
        max_agents: int | None = 11,
        map_size: int = MAP_SIZE,
        sent_dim: int = 384,
        sbert_model: str = "all-MiniLM-L6-v2",
        val_frac: float = 0.1,
        use_metres: bool = True,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.max_agents = max_agents
        self.target_dt  = target_dt
        self.sent_dim   = sent_dim

        nba_root  = Path(nba_root)
        cache_dir = nba_root / "cache" / "ctx"
        M         = build_court_map(map_size)
        sbert     = _try_load_sbert(sbert_model)
        C         = _embed(NBA_DESCRIPTION, sent_dim, sbert, cache_dir)

        # ── Load data file ────────────────────────────────────────────────
        if split == "test":
            npy_path = nba_root / "test.npy"
        else:
            npy_path = nba_root / "train.npy"

        assert npy_path.exists(), (
            f"[NBA] {npy_path} not found.\n"
            "Download from https://github.com/linouk23/NBA-Player-Movements\n"
            "or use the preprocessed version from GroupNet / EvolveGraph repos."
        )

        sequences = load_nba_npy(npy_path, use_metres=use_metres)

        windows = _extract_windows_npy(
            sequences,
            seq_len=self.seq_len,
            target_dt=target_dt,
            raw_dt=DT_25HZ,
            max_agents=max_agents,
        )

        # ── Train / val split ─────────────────────────────────────────────
        n     = len(windows)
        n_val = max(1, int(n * val_frac))
        n_tr  = n - n_val
        if split == "train":
            windows = windows[:n_tr]
        elif split == "val":
            windows = windows[n_tr:]

        assert windows, f"[NBA] No windows for split='{split}'"
        self._windows = windows
        self._M       = M
        self._C       = C
        print(f"[NBA] split={split:5s}  windows={len(windows):6d}  agents≤{max_agents}")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        traj_np = self._windows[idx]                          # (N, T, 2) metres
        tau0    = torch.from_numpy(traj_np).float()

        if self.max_agents is not None:
            N = tau0.shape[0]
            if N < self.max_agents:
                pad  = torch.zeros(self.max_agents - N, self.seq_len, 2)
                tau0 = torch.cat([tau0, pad], dim=0)

        pos0 = tau0[:, 0, :]
        vel0 = (tau0[:, 1, :] - pos0) / self.target_dt
        goal = tau0[:, -1, :]
        S0   = torch.cat([pos0, vel0, goal], dim=-1)

        return tau0, self._M, self._C, S0
