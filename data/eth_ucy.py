"""
ETH/UCY Pedestrian Dataset Loader for CRDTraj.

Scenes (5 standard leave-one-out subsets):
  eth   — ETH Zurich outdoor plaza          (biwi_eth.txt)
  hotel — ETH hotel entrance area           (biwi_hotel.txt)
  univ  — UCY university campus             (students001.txt + students003.txt)
  zara1 — UCY Zara shopping street scene 1  (crowds_zara01.txt)
  zara2 — UCY Zara shopping street scene 2  (crowds_zara02.txt)

All coordinates are in METRES.  Frame interval: 0.4 s (25 fps sampled every 10).

Data is downloaded automatically from the Social-STGCNN GitHub mirror on first use:
  https://github.com/abduallahmohamed/Social-STGCNN

Each __getitem__ returns (tau0, M, C, S0):
  tau0 : (N, T, 2)     full trajectory in metres,     float32
  M    : (3, H, W)     top-down occupancy map [0, 1], float32
  C    : (sent_dim,)   SBERT scene embedding,          float32
  S0   : (N, 6)        initial agent state,            float32
                       = [x₀, y₀, vx₀, vy₀, gx, gy]
                         positions in m, velocities in m/s,
                         goal = trajectory endpoint
"""

import hashlib
import math
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.4          # seconds between consecutive preprocessed frames (25 fps / skip 10)
MAP_SIZE = 224    # pixels per side for the occupancy map

# Human-readable scene descriptions → fed to SentenceBERT
SCENE_DESCRIPTIONS: dict[str, str] = {
    "eth":   "Pedestrians walking outdoors on the ETH Zurich university campus plaza",
    "hotel": "Pedestrians navigating through a hotel outdoor entrance and walkway area",
    "univ":  "Students and visitors crossing an open university campus courtyard",
    "zara1": "Pedestrians and shoppers walking along a busy commercial street",
    "zara2": "Pedestrians and shoppers moving along a commercial street, second sequence",
}

# Base URL for all raw trajectory files
_GITHUB_RAW = (
    "https://raw.githubusercontent.com/abduallahmohamed/Social-STGCNN/master/datasets/raw/all_data"
)

# Per-scene raw txt files (all already in metres, tab-separated: frame ped x y)
# univ uses two recording sessions; we merge them as one scene.
_SCENE_FILES: dict[str, list[str]] = {
    "eth":   ["biwi_eth.txt"],
    "hotel": ["biwi_hotel.txt"],
    "univ":  ["students001.txt", "students003.txt"],
    "zara1": ["crowds_zara01.txt"],
    "zara2": ["crowds_zara02.txt"],
}

# Standard leave-one-out benchmark splits
LOO_SPLITS: dict[str, dict[str, list[str]]] = {
    "eth":   {"test": ["eth"],   "train": ["hotel", "univ", "zara1", "zara2"]},
    "hotel": {"test": ["hotel"], "train": ["eth",   "univ", "zara1", "zara2"]},
    "univ":  {"test": ["univ"],  "train": ["eth",   "hotel", "zara1", "zara2"]},
    "zara1": {"test": ["zara1"], "train": ["eth",   "hotel", "univ",  "zara2"]},
    "zara2": {"test": ["zara2"], "train": ["eth",   "hotel", "univ",  "zara1"]},
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ETH/UCY] Downloading {dest.name} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed: {url}\n"
            f"Place the file manually at: {dest}\n"
            f"Original error: {exc}"
        ) from exc


def download_all_scenes(root: Path) -> dict[str, list[Path]]:
    """
    Download every raw scene file if not already present.

    Returns
    -------
    dict  scene_name → list of local Path objects
    """
    raw_dir = root / "raw"
    scene_paths: dict[str, list[Path]] = {}
    for scene, fnames in _SCENE_FILES.items():
        paths = []
        for fname in fnames:
            dest = raw_dir / fname
            _download(f"{_GITHUB_RAW}/{fname}", dest)
            paths.append(dest)
        scene_paths[scene] = paths
    return scene_paths


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def _parse_txt(path: Path) -> np.ndarray:
    """
    Parse a whitespace-separated trajectory file.

    Each line: frame_id  ped_id  x  y   (tab or space, floats)
    Returns ndarray (M, 4) float32: [frame_id, ped_id, x_m, y_m]
    """
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append([float(p) for p in parts[:4]])
    return np.array(rows, dtype=np.float32)


def load_scene(paths: list[Path]) -> np.ndarray:
    """
    Load and concatenate raw files for one scene.

    When a scene has multiple recording files (e.g. univ = students001 + students003)
    we offset frame IDs and pedestrian IDs so they don't collide.

    Returns
    -------
    np.ndarray  shape (M, 4): [frame_id, ped_id, x_m, y_m]
    """
    chunks: list[np.ndarray] = []
    frame_offset = 0
    ped_offset = 0
    for path in paths:
        data = _parse_txt(path)
        if data.size == 0:
            continue
        # Zero-base frame IDs within this file
        data[:, 0] -= data[:, 0].min()
        # Apply offsets so no collision with previous files
        data[:, 0] += frame_offset
        data[:, 1] += ped_offset
        frame_offset = int(data[:, 0].max()) + 1
        ped_offset = int(data[:, 1].max()) + 1
        chunks.append(data)
    if not chunks:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Sliding-window sequence extraction
# ---------------------------------------------------------------------------

def extract_sequences(
    data: np.ndarray,
    seq_len: int,
    min_agents: int = 2,
    max_agents: int | None = None,
    stride: int = 1,
) -> list[np.ndarray]:
    """
    Slide a window of `seq_len` consecutive frames over the scene.

    Only agents present in **all** frames of the window are kept.
    Windows with fewer than `min_agents` complete tracks are dropped.

    Parameters
    ----------
    data       : (M, 4) [frame_id, ped_id, x, y]
    seq_len    : number of consecutive frames per sequence (obs + pred)
    min_agents : minimum number of complete tracks to keep a window
    max_agents : if set, truncate agent list to this size (sorted by ID)
    stride     : step between consecutive windows (default 1 = fully overlapping)

    Returns
    -------
    list of (N_i, seq_len, 2) float32 arrays  — positions in metres
    """
    frame_ids = np.unique(data[:, 0]).astype(int)
    frame_ids.sort()
    n_frames = len(frame_ids)
    if n_frames < seq_len:
        return []

    # frame_id → {ped_id: (x, y)}
    frame_dict: dict[int, dict[int, tuple[float, float]]] = {}
    for row in data:
        fid, pid, x, y = int(row[0]), int(row[1]), float(row[2]), float(row[3])
        frame_dict.setdefault(fid, {})[pid] = (x, y)

    sequences: list[np.ndarray] = []
    for start in range(0, n_frames - seq_len + 1, stride):
        window = frame_ids[start : start + seq_len]

        # Agents present in ALL frames of the window
        present = set(frame_dict.get(window[0], {}).keys())
        for fid in window[1:]:
            present &= set(frame_dict.get(fid, {}).keys())

        if len(present) < min_agents:
            continue

        agents = sorted(present)
        if max_agents is not None:
            agents = agents[:max_agents]

        # Build (N, seq_len, 2)
        traj = np.array(
            [
                [frame_dict[fid][pid] for fid in window]
                for pid in agents
            ],
            dtype=np.float32,
        )  # (N, seq_len, 2)
        sequences.append(traj)

    return sequences


# ---------------------------------------------------------------------------
# Occupancy map generation (no external images required)
# ---------------------------------------------------------------------------

class SceneExtent:
    """World-coordinate bounding box for a scene."""

    def __init__(self, data: np.ndarray, margin: float = 2.0):
        xs, ys = data[:, 2], data[:, 3]
        self.x_min = float(xs.min()) - margin
        self.x_max = float(xs.max()) + margin
        self.y_min = float(ys.min()) - margin
        self.y_max = float(ys.max()) + margin
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

    def world_to_px(
        self, x: np.ndarray, y: np.ndarray, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert world (metres) → pixel indices in a (size × size) grid."""
        px = np.clip((x - self.x_min) / self.x_range * (size - 1), 0, size - 1).astype(int)
        py = np.clip((y - self.y_min) / self.y_range * (size - 1), 0, size - 1).astype(int)
        return px, py

    def metres_per_pixel(self, size: int) -> tuple[float, float]:
        return self.x_range / size, self.y_range / size


def _gaussian_blur_torch(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Simple separable Gaussian blur implemented in PyTorch (no scipy needed)."""
    if sigma <= 0:
        return grid
    # Kernel radius
    radius = max(1, int(math.ceil(3 * sigma)))
    ks = 2 * radius + 1
    t = torch.arange(ks, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    # Separable 2-D convolution
    k2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (ks, ks)
    k2d = k2d.view(1, 1, ks, ks)
    g = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pad = radius
    blurred = torch.nn.functional.conv2d(g, k2d, padding=pad)
    return blurred.squeeze(0).squeeze(0)


def build_occupancy_map(
    data: np.ndarray,
    extent: SceneExtent,
    size: int = MAP_SIZE,
    sigma_px: float = 3.0,
) -> torch.Tensor:
    """
    Build a top-down occupancy/walkability map from pedestrian trajectories.

    Strategy
    --------
    1. Accumulate visit counts on a (size × size) pixel grid.
    2. Apply a Gaussian blur so unvisited-but-adjacent cells get nonzero weight.
    3. Normalise to [0, 1]:  1.0 = heavily trafficked / known walkable,
                             0.0 = unobserved (potential obstacle or out-of-scene).

    The map is a floating-point grayscale image replicated to 3 channels so it
    can be fed directly to the ResNet-18 MapEncoder.  Values are in [0, 1]
    (NOT ImageNet-normalised — the encoder receives raw [0, 1] input).

    Parameters
    ----------
    data      : (M, 4) full-scene trajectory array [frame_id, ped_id, x_m, y_m]
    extent    : SceneExtent bounding box
    size      : pixel side length of the output map (default 224)
    sigma_px  : Gaussian smoothing radius in pixels

    Returns
    -------
    (3, size, size) float32 tensor in [0, 1]
    """
    grid = np.zeros((size, size), dtype=np.float32)
    px, py = extent.world_to_px(data[:, 2], data[:, 3], size)
    np.add.at(grid, (py, px), 1.0)

    # Gaussian smoothing (torch — no scipy dependency)
    grid_t = torch.from_numpy(grid)
    grid_t = _gaussian_blur_torch(grid_t, sigma=sigma_px)

    # Normalise to [0, 1]
    vmax = grid_t.max()
    if vmax > 0:
        grid_t = grid_t / vmax

    # Replicate to 3 channels
    return grid_t.unsqueeze(0).expand(3, -1, -1).contiguous()  # (3, H, W) ∈ [0,1]


# ---------------------------------------------------------------------------
# Context embedding (SentenceBERT, cached)
# ---------------------------------------------------------------------------

def _embed_description(
    text: str,
    sent_dim: int,
    sbert,          # SentenceTransformer instance or None
    cache_dir: Path,
) -> torch.Tensor:
    """
    Compute (or load from cache) the SentenceBERT embedding for `text`.

    Falls back to a zero vector if sentence_transformers is unavailable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.pt"

    if cache_file.exists():
        return torch.load(str(cache_file), map_location="cpu", weights_only=True)

    if sbert is not None:
        with torch.no_grad():
            emb = sbert.encode(text, convert_to_tensor=True).cpu().float()
    else:
        emb = torch.zeros(sent_dim)

    torch.save(emb, str(cache_file))
    return emb


# ---------------------------------------------------------------------------
# ETHUCYDataset
# ---------------------------------------------------------------------------

class ETHUCYDataset(Dataset):
    """
    ETH/UCY pedestrian trajectory dataset formatted for CRDTraj.

    Leave-one-out benchmark
    -----------------------
    Five scenes; one is held out as the test scene and the remaining four
    are used for training.  Pass ``test_scene`` to choose the held-out scene.

    Data layout
    -----------
    Raw files are downloaded on first use from the Social-STGCNN GitHub mirror
    and cached under ``root/raw/``.  Context embeddings are cached under
    ``root/cache/ctx/``.

    Sample contents (N = number of agents in window)
    -------------------------------------------------
    tau0 : (N, T, 2)     full trajectory (obs + pred) in metres     float32
    M    : (3, H, W)     top-down occupancy map in [0, 1]           float32
    C    : (sent_dim,)   frozen SBERT embedding of scene text        float32
    S0   : (N, 6)        initial agent state [x₀,y₀,vx₀,vy₀,gx,gy] float32

    Parameters
    ----------
    root           : directory for raw files and caches
    split          : 'train' | 'val' | 'test'
    test_scene     : held-out scene for the LOO benchmark (default 'eth')
    obs_len        : observed frames (default 8 → 3.2 s)
    pred_len       : future frames to predict (default 12 → 4.8 s)
    min_agents     : drop windows with fewer agents (default 2)
    max_agents     : pad/truncate all sequences to this agent count;
                     None keeps the natural per-window count (requires
                     a custom collate_fn for batching)
    map_size       : occupancy map pixel side (default 224)
    map_sigma      : Gaussian smoothing radius in pixels (default 3.0)
    sent_dim       : SBERT embedding dimension (default 384 for all-MiniLM-L6-v2)
    sbert_model    : SentenceTransformer model name
    val_frac       : fraction of TRAINING sequences held back for validation
    stride         : window stride (default 1 = fully overlapping sequences)
    normalize_traj : if True, translate each sequence so the mean of
                     agent positions at t=0 is the origin (default False)
    """

    SCENE_NAMES = ["eth", "hotel", "univ", "zara1", "zara2"]

    def __init__(
        self,
        root: str = "data/eth_ucy",
        split: Literal["train", "val", "test"] = "train",
        test_scene: str = "eth",
        obs_len: int = 8,
        pred_len: int = 12,
        min_agents: int = 2,
        max_agents: int | None = None,
        map_size: int = MAP_SIZE,
        map_sigma: float = 3.0,
        sent_dim: int = 384,
        sbert_model: str = "all-MiniLM-L6-v2",
        val_frac: float = 0.1,
        stride: int = 1,
        normalize_traj: bool = False,
    ):
        super().__init__()
        assert test_scene in self.SCENE_NAMES, (
            f"test_scene must be one of {self.SCENE_NAMES}, got '{test_scene}'"
        )
        assert split in ("train", "val", "test"), f"split must be train/val/test, got '{split}'"

        self.root = Path(root)
        self.split = split
        self.test_scene = test_scene
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.map_size = map_size
        self.map_sigma = map_sigma
        self.sent_dim = sent_dim
        self.dt = DT
        self.normalize_traj = normalize_traj

        # ── Download raw files ──────────────────────────────────────────────
        scene_paths = download_all_scenes(self.root)

        # ── Load SentenceBERT (optional) ────────────────────────────────────
        sbert = self._try_load_sbert(sbert_model)

        # ── Which scenes to load for this split ─────────────────────────────
        if split == "test":
            active_scenes = LOO_SPLITS[test_scene]["test"]
        else:
            active_scenes = LOO_SPLITS[test_scene]["train"]

        # ── Build per-scene maps and context embeddings ──────────────────────
        # The occupancy map is built from ALL scene trajectories (test included)
        # so the model sees a complete scene layout even at inference time.
        self._scene_raw: dict[str, np.ndarray] = {}
        self._scene_extent: dict[str, SceneExtent] = {}
        self._scene_map: dict[str, torch.Tensor] = {}
        self._scene_ctx: dict[str, torch.Tensor] = {}

        for scene in self.SCENE_NAMES:
            raw = load_scene(scene_paths[scene])
            if raw.size == 0:
                continue
            self._scene_raw[scene] = raw
            ext = SceneExtent(raw, margin=2.0)
            self._scene_extent[scene] = ext
            self._scene_map[scene] = build_occupancy_map(raw, ext, size=map_size, sigma_px=map_sigma)
            self._scene_ctx[scene] = _embed_description(
                SCENE_DESCRIPTIONS[scene],
                sent_dim,
                sbert,
                self.root / "cache" / "ctx",
            )

        # ── Extract sliding-window samples ──────────────────────────────────
        all_samples: list[tuple[str, np.ndarray]] = []  # (scene, traj)
        for scene in active_scenes:
            raw = self._scene_raw.get(scene)
            if raw is None or raw.size == 0:
                continue
            seqs = extract_sequences(
                raw,
                seq_len=self.seq_len,
                min_agents=min_agents,
                max_agents=max_agents,
                stride=stride,
            )
            for traj in seqs:
                all_samples.append((scene, traj))

        # ── Train / val temporal split ───────────────────────────────────────
        if split in ("train", "val"):
            n_val = max(1, int(len(all_samples) * val_frac))
            n_train = len(all_samples) - n_val
            if split == "train":
                self._samples = all_samples[:n_train]
            else:
                self._samples = all_samples[n_train:]
        else:
            self._samples = all_samples

        if not self._samples:
            raise RuntimeError(
                f"No sequences found for split='{split}', test_scene='{test_scene}'. "
                "Check that files downloaded correctly under "
                f"{self.root / 'raw'}."
            )

        print(
            f"[ETH/UCY] split={split:5s}  test_scene={test_scene}  "
            f"sequences={len(self._samples):5d}"
        )

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scene, traj_np = self._samples[idx]

        tau0 = torch.from_numpy(traj_np).float()  # (N, T, 2)  metres

        # Optional coordinate normalisation: centre on mean start position
        if self.normalize_traj:
            origin = tau0[:, 0, :].mean(dim=0, keepdim=True).unsqueeze(1)  # (1, 1, 2)
            tau0 = tau0 - origin

        # Pad / truncate agents if needed
        if self.max_agents is not None:
            N = tau0.shape[0]
            if N > self.max_agents:
                tau0 = tau0[: self.max_agents]
            elif N < self.max_agents:
                pad = torch.zeros(self.max_agents - N, self.seq_len, 2)
                tau0 = torch.cat([tau0, pad], dim=0)

        M = self._scene_map[scene]   # (3, H, W)  [0, 1]
        C = self._scene_ctx[scene]   # (sent_dim,)
        S0 = self._compute_s0(tau0)  # (N, 6)

        return tau0, M, C, S0

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_s0(self, tau0: torch.Tensor) -> torch.Tensor:
        """
        Compute initial agent state s₀^i = (x₀, y₀, vx₀, vy₀, gx, gy).

          (x₀, y₀)   — position at t = 0  (metres)
          (vx₀, vy₀) — velocity at t = 0  (m/s, finite difference)
          (gx, gy)    — goal = trajectory endpoint at t = T−1  (metres)
        """
        pos0 = tau0[:, 0, :]                      # (N, 2)
        vel0 = (tau0[:, 1, :] - pos0) / self.dt   # (N, 2)  m/s
        goal = tau0[:, -1, :]                      # (N, 2)
        return torch.cat([pos0, vel0, goal], dim=-1)  # (N, 6)

    @staticmethod
    def _try_load_sbert(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            print(
                "[ETH/UCY] Warning: sentence-transformers not installed. "
                "Context embeddings will be zero vectors."
            )
            return None

    # ── Metadata ─────────────────────────────────────────────────────────────

    def scene_extent(self, scene: str) -> SceneExtent:
        """World-coordinate bounding box for a scene."""
        return self._scene_extent[scene]

    def metres_per_pixel(self, scene: str) -> tuple[float, float]:
        """(dx, dy) metres per map pixel for a given scene."""
        return self._scene_extent[scene].metres_per_pixel(self.map_size)

    @property
    def active_scenes(self) -> list[str]:
        if self.split == "test":
            return LOO_SPLITS[self.test_scene]["test"]
        return LOO_SPLITS[self.test_scene]["train"]


# ---------------------------------------------------------------------------
# Custom collate_fn for variable-agent batches
# ---------------------------------------------------------------------------

def ethucy_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a batch of (tau0, M, C, S0) samples where tau0 and S0 may have
    different numbers of agents.

    Pads all sequences in the batch to the maximum agent count found, filling
    padded positions with zeros.

    Returns
    -------
    tau0 : (B, N_max, T, 2)
    M    : (B, 3, H, W)
    C    : (B, sent_dim)
    S0   : (B, N_max, 6)
    """
    tau0_list, M_list, C_list, S0_list = zip(*batch)

    N_max = max(t.shape[0] for t in tau0_list)
    T = tau0_list[0].shape[1]

    tau0_padded = torch.zeros(len(batch), N_max, T, 2)
    S0_padded = torch.zeros(len(batch), N_max, 6)

    for i, (tau0, s0) in enumerate(zip(tau0_list, S0_list)):
        N = tau0.shape[0]
        tau0_padded[i, :N] = tau0
        S0_padded[i, :N] = s0

    return (
        tau0_padded,
        torch.stack(M_list),
        torch.stack(C_list),
        S0_padded,
    )
