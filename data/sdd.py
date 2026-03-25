# obs_len / pred_len split is only for benchmark eval — training uses the full seq_len window
"""
Stanford Drone Dataset (SDD) dataloader.

Layout expected
---------------
    sdd_root/
        annotation/
            {scene}/
                {video}/
                    annotations.txt    (track_id frame_id x1 y1 x2 y2 lost occ gen label)
                    reference.jpg      (aerial photo, same resolution as seg map)
        segmentation/
            {scene}_{video_num}_seg.png   (semantic segmentation, uint8)
        homography/
            estimated_scales.yaml         (metres-per-pixel per scene/video)

Map
---
The segmentation map is used as the scene obstacle map:
  - Walkable labels  (2, 4, 5, 6, 7): pedestrians observed here  → 1.0
  - Obstacle labels  (1, 3)          : buildings / fences          → 0.0
Returns (3, 224, 224) float32 tensor, same format as ETH/UCY / Synthetic.

Coordinates
-----------
Bounding-box centres in pixels × scale → metres.
Scale is loaded from estimated_scales.yaml.

Scenes
------
bookstore, coupa, deathCircle, gates, hyang, little, nexus, quad
"""

import hashlib
from pathlib import Path
from typing import Literal

from data.cache import DatasetCache, cache_name, DEFAULT_CACHE_DIR

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAP_SIZE     = 224
SEQ_LEN      = 20
DT           = 0.4      # seconds per frame after 2.5 fps subsampling (25fps / 10)
RAW_FPS      = 30       # SDD raw frame rate (approx)
TARGET_FPS   = 2.5      # standard SDD evaluation frequency

# Segmentation labels: which pixel values count as walkable
# Empirically determined by overlaying pedestrian trajectories with the seg map.
# Label 1 = buildings/obstacles, label 3 = fences/obstacles
WALKABLE_LABELS = {2, 4, 5, 6, 7}
OBSTACLE_LABELS = {1, 3}

SCENE_DESCRIPTIONS = {
    "bookstore":   "Pedestrians near an outdoor bookstore on Stanford University campus",
    "coupa":       "Students and visitors around Coupa Café on Stanford University campus",
    "deathCircle": "Pedestrians navigating a circular plaza intersection on Stanford campus",
    "gates":       "Students walking near the Gates Computer Science building at Stanford",
    "hyang":       "Pedestrians crossing a busy intersection near Huang Engineering at Stanford",
    "little":      "Pedestrians around Little outdoor eating area on Stanford campus",
    "nexus":       "Students and staff crossing the Nexus courtyard at Stanford University",
    "quad":        "Pedestrians walking through the Main Quad at Stanford University",
}

ALL_SCENES = list(SCENE_DESCRIPTIONS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scales(sdd_root: Path) -> dict[str, dict[str, float]]:
    """
    Return {scene: {video_dir: scale_m_per_px}}.
    Requires PyYAML.  Falls back to a fixed default (0.038 m/px) if unavailable.
    """
    yaml_path = sdd_root / "homography" / "estimated_scales.yaml"
    if not _YAML or not yaml_path.exists():
        default = 0.038
        return {s: {f"video{i}": default for i in range(20)} for s in ALL_SCENES}

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    scales: dict[str, dict[str, float]] = {}
    for scene, vids in raw.items():
        scales[scene] = {}
        for vid_key, info in vids.items():
            scales[scene][vid_key] = float(info["scale"])
    return scales


def _load_seg_map(sdd_root: Path, scene: str, video_idx: int, size: int = MAP_SIZE) -> torch.Tensor:
    """
    Load {scene}_{video_idx}_seg.png, convert to binary walkable map,
    resize to (size, size), return (3, size, size) float32 tensor in [0, 1].

    1.0 = walkable  (pedestrians can be here)
    0.0 = obstacle  (buildings, fences)
    """
    seg_path = sdd_root / "segmentation" / f"{scene}_{video_idx}_seg.png"
    if not seg_path.exists() or not _PIL:
        # Fallback: all-walkable map
        return torch.ones(3, size, size)

    seg = np.array(Image.open(seg_path))             # (H, W) uint8
    walkable = np.zeros_like(seg, dtype=np.float32)
    for lbl in WALKABLE_LABELS:
        walkable[seg == lbl] = 1.0

    # Resize via torch (nearest neighbour to preserve binary values)
    t = torch.from_numpy(walkable).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    t = F.interpolate(t, size=(size, size), mode="nearest")    # (1,1,S,S)
    t = t.squeeze()                                             # (S, S)
    return t.unsqueeze(0).expand(3, -1, -1).contiguous()


def _load_annotations(ann_path: Path, scale: float) -> np.ndarray | None:
    """
    Parse annotations.txt.  Returns float32 array (N_obs, 4) = [frame_id, ped_id, x_m, y_m]
    for pedestrians only (label == "Pedestrian"), excluding lost frames.

    Subsampled to ~2.5 fps (every 12 frames from 30 fps raw).
    """
    records = []
    subsample = round(RAW_FPS / TARGET_FPS)   # = 12

    # SDD annotation format (from original README):
    #   track_id  xmin  ymin  xmax  ymax  frame  lost  occluded  generated  label
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            track_id  = int(parts[0])
            x1, y1    = int(parts[1]), int(parts[2])
            x2, y2    = int(parts[3]), int(parts[4])
            frame_id  = int(parts[5])
            lost      = int(parts[6])
            label     = parts[9].strip('"')

            if label != "Pedestrian" or lost == 1:
                continue
            if frame_id % subsample != 0:
                continue

            cx = (x1 + x2) / 2.0 * scale
            cy = (y1 + y2) / 2.0 * scale
            records.append((frame_id // subsample, track_id, cx, cy))

    if not records:
        return None
    return np.array(records, dtype=np.float32)   # (N, 4)


def _extract_windows(
    records: np.ndarray,
    seq_len: int,
    stride: int,
    min_agents: int,
    max_agents: int | None,
) -> list[np.ndarray]:
    """
    Sliding-window extraction from (N, 4) = [sub_frame, ped_id, x, y].
    Returns list of (N_agents, seq_len, 2) arrays in metres.
    """
    frames = np.unique(records[:, 0]).astype(int)
    if len(frames) < seq_len:
        return []

    frame_to_peds: dict[int, dict[int, tuple[float, float]]] = {}
    for row in records:
        fid, pid, x, y = int(row[0]), int(row[1]), float(row[2]), float(row[3])
        frame_to_peds.setdefault(fid, {})[pid] = (x, y)

    # Convert to dense frame index
    frame_arr = sorted(frame_to_peds.keys())
    f2i = {f: i for i, f in enumerate(frame_arr)}
    T = len(frame_arr)

    windows = []
    for start in range(0, T - seq_len + 1, stride):
        window_frames = frame_arr[start: start + seq_len]
        # Pedestrians present in ALL frames of the window
        present = set(frame_to_peds[window_frames[0]].keys())
        for f in window_frames[1:]:
            present &= set(frame_to_peds[f].keys())

        if len(present) < min_agents:
            continue

        peds = sorted(present)
        if max_agents is not None:
            peds = peds[:max_agents]
        N = len(peds)

        traj = np.zeros((N, seq_len, 2), dtype=np.float32)
        for t, f in enumerate(window_frames):
            for n, pid in enumerate(peds):
                traj[n, t] = frame_to_peds[f][pid]
        windows.append(traj)

    return windows


def _embed(text: str, sent_dim: int, sbert, cache_dir: Path) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(text.encode()).hexdigest()[:16]
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
        print("[SDD] Warning: sentence-transformers not available; using zero context.")
        return None


# ---------------------------------------------------------------------------
# SDDDataset
# ---------------------------------------------------------------------------

class SDDDataset(Dataset):
    """
    Stanford Drone Dataset.

    Each sample returns (tau0, M, C, S0):
      tau0 : (N, T, 2)        trajectory in metres    float32
      M    : (3, 224, 224)    seg-map obstacle map     float32  (1=walkable, 0=obstacle)
      C    : (sent_dim,)      SBERT scene embedding    float32
      S0   : (N, 6)           [x0,y0,vx0,vy0,gx,gy]  float32

    Parameters
    ----------
    sdd_root    : path to SDD root (contains annotation/, segmentation/, homography/)
    scenes      : list of scene names to include (None = all 8 scenes)
    split       : 'train' | 'val' | 'test'
    test_scenes : scenes held out for testing (leave-one-out); None uses val_frac only
    seq_len     : total trajectory window length in frames
    min_agents  : minimum co-present pedestrians
    max_agents  : pad/truncate agents (None = variable)
    map_size    : output map resolution (pixels)
    sent_dim    : SBERT dimension
    sbert_model : SentenceTransformer model name
    val_frac    : fraction of each scene's windows used for validation
    stride      : window extraction stride (default = seq_len)
    """

    def __init__(
        self,
        sdd_root: str,
        scenes: list[str] | None = None,
        split: Literal["train", "val", "test"] = "train",
        test_scenes: list[str] | None = None,
        seq_len: int = SEQ_LEN,
        min_agents: int = 2,
        max_agents: int | None = None,
        map_size: int = MAP_SIZE,
        sent_dim: int = 384,
        sbert_model: str = "all-MiniLM-L6-v2",
        val_frac: float = 0.1,
        stride: int | None = None,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.max_agents = max_agents
        self.sent_dim   = sent_dim
        self.dt         = DT

        sdd_root   = Path(sdd_root)
        _stride    = stride if stride is not None else self.seq_len
        ann_root   = sdd_root / "annotation"
        active     = scenes if scenes is not None else ALL_SCENES
        test_set   = set(test_scenes) if test_scenes else set()

        if split == "test":
            active = [s for s in active if s in test_set] or active
        elif split in ("train", "val"):
            active = [s for s in active if s not in test_set]

        _cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        _cname = cache_name(
            dataset="sdd", split=split,
            scenes=sorted(active), test=sorted(test_set),
            seq=seq_len, stride=_stride,
            N=max_agents or "var", valf=val_frac,
        )
        disk_cache = DatasetCache(_cache_root, _cname)

        if disk_cache.exists():
            self._samples = disk_cache.load()
            print(f"[SDD] loaded from cache  split={split:5s}  windows={len(self._samples):5d}")
            return

        scales = _load_scales(sdd_root)
        sbert  = _try_load_sbert(sbert_model)
        ctx_dir = sdd_root / "cache" / "ctx"

        self._samples: list[tuple[np.ndarray, torch.Tensor, torch.Tensor]] = []

        for scene in active:
            scene_dir = ann_root / scene
            if not scene_dir.exists():
                continue

            ctx = SCENE_DESCRIPTIONS.get(scene, f"Pedestrians at {scene}")
            C   = _embed(ctx, sent_dim, sbert, ctx_dir)

            for video_dir in sorted(scene_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                ann_path = video_dir / "annotations.txt"
                if not ann_path.exists():
                    continue

                video_name = video_dir.name          # "video0", "video1", ...
                video_idx  = int(video_name.replace("video", ""))
                scale      = scales.get(scene, {}).get(video_name, 0.038)

                records = _load_annotations(ann_path, scale)
                if records is None:
                    continue

                M = _load_seg_map(sdd_root, scene, video_idx, map_size)

                windows = _extract_windows(
                    records,
                    seq_len=self.seq_len,
                    stride=_stride,
                    min_agents=min_agents,
                    max_agents=max_agents,
                )
                for w in windows:
                    self._samples.append((w, M, C))

        # Train / val split (within the active scenes)
        n     = len(self._samples)
        n_val = max(1, int(n * val_frac))
        n_tr  = n - n_val
        if split == "train":
            self._samples = self._samples[:n_tr]
        elif split == "val":
            self._samples = self._samples[n_tr:]

        assert self._samples, f"[SDD] No windows found for split='{split}', scenes={active}"
        print(
            f"[SDD] split={split:5s}  scenes={active}  "
            f"windows={len(self._samples):5d}"
        )
        disk_cache.save(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        traj_np, M, C = self._samples[idx]
        tau0 = torch.from_numpy(traj_np).float()

        if self.max_agents is not None:
            N = tau0.shape[0]
            if N > self.max_agents:
                tau0 = tau0[: self.max_agents]
            elif N < self.max_agents:
                pad = torch.zeros(self.max_agents - N, self.seq_len, 2)
                tau0 = torch.cat([tau0, pad], dim=0)

        S0 = self._compute_s0(tau0)
        return tau0, M, C, S0

    def _compute_s0(self, tau0: torch.Tensor) -> torch.Tensor:
        pos0 = tau0[:, 0, :]
        vel0 = (tau0[:, 1, :] - pos0) / self.dt
        goal = tau0[:, -1, :]
        return torch.cat([pos0, vel0, goal], dim=-1)


# ---------------------------------------------------------------------------
# GCS (Grand Central Station) dataset
# ---------------------------------------------------------------------------

"""
GCS format
----------
  annotation/  {000001..012684}.txt — one file per frame
      each file: N rows of 3 whitespace-separated values: x_pixel  y_pixel  ped_id
  homography/  terminal_H.txt — 3×3 homography matrix (pixel → world metres)
  segmentation/ terminal_seg.png
  reference/    terminal_image.jpg

Homography usage:  [X_m, Y_m, 1]^T = H @ [x_px, y_px, 1]^T  (then divide by last coord)
"""

GCS_DESCRIPTION = "Dense crowd of commuters moving through Grand Central Terminal, New York"


def _load_homography(h_path: Path) -> np.ndarray:
    """Load a 3×3 homography matrix from a whitespace-separated text file."""
    rows = []
    with open(h_path) as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                rows.append([float(v) for v in vals])
    return np.array(rows, dtype=np.float64)


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply 3×3 homography to (N, 2) pixel points.
    Returns (N, 2) world coordinates in metres.
    """
    ones  = np.ones((len(pts), 1), dtype=np.float64)
    hpts  = np.hstack([pts, ones])          # (N, 3)
    world = (H @ hpts.T).T                  # (N, 3)
    world = world[:, :2] / world[:, 2:3]    # (N, 2) — divide by homogeneous coord
    return world.astype(np.float32)


def _load_gcs_seg_map(gcs_root: Path, size: int = MAP_SIZE) -> torch.Tensor:
    seg_path = gcs_root / "segmentation" / "terminal_seg.png"
    if not seg_path.exists() or not _PIL:
        return torch.ones(3, size, size)

    seg = np.array(Image.open(seg_path))
    # Labels 1,2,6 found in GCS.  Cross-validate with ped positions:
    # use labels 2,6 as walkable (open floor areas), 1 as obstacle (pillars/walls)
    walkable = np.isin(seg, [2, 6]).astype(np.float32)
    t = torch.from_numpy(walkable).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="nearest").squeeze()
    return t.unsqueeze(0).expand(3, -1, -1).contiguous()


class GCSDataset(Dataset):
    """
    Grand Central Station pedestrian dataset.

    Each sample returns (tau0, M, C, S0) in the same format as SDDDataset.

    Parameters
    ----------
    gcs_root  : path containing annotation/, homography/, segmentation/
    split     : 'train' | 'val' | 'test'
    seq_len   : total trajectory window length in frames (default 20)
    target_dt : desired time step in seconds (raw dt ≈ 0.4 s at 2.5 fps assumed)
    """

    def __init__(
        self,
        gcs_root: str,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = SEQ_LEN,
        target_dt: float = DT,
        min_agents: int = 2,
        max_agents: int | None = None,
        map_size: int = MAP_SIZE,
        sent_dim: int = 384,
        sbert_model: str = "all-MiniLM-L6-v2",
        val_frac: float = 0.1,
        stride: int | None = None,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.max_agents = max_agents
        self.sent_dim   = sent_dim
        self.dt         = target_dt

        gcs_root  = Path(gcs_root)
        _stride   = stride if stride is not None else self.seq_len

        _cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        _cname = cache_name(
            dataset="gcs", split=split,
            seq=seq_len, stride=_stride,
            N=max_agents or "var", valf=val_frac,
        )
        disk_cache = DatasetCache(_cache_root, _cname)

        H = _load_homography(gcs_root / "homography" / "terminal_H.txt")
        M = _load_gcs_seg_map(gcs_root, map_size)

        sbert   = _try_load_sbert(sbert_model)
        ctx_dir = gcs_root / "cache" / "ctx"
        C       = _embed(GCS_DESCRIPTION, sent_dim, sbert, ctx_dir)

        if disk_cache.exists():
            self._windows = disk_cache.load()
            self._M = M
            self._C = C
            print(f"[GCS] loaded from cache  split={split:5s}  windows={len(self._windows):5d}")
            return

        # ── Load all frames ────────────────────────────────────────────────
        ann_dir = gcs_root / "annotation"
        # GCS: frame files are named 000001.txt … 012684.txt
        frame_files = sorted(ann_dir.glob("*.txt"))
        frame_data: dict[int, dict[int, np.ndarray]] = {}   # {frame: {ped_id: (x,y)}}

        for ff in frame_files:
            try:
                fid = int(ff.stem)
            except ValueError:
                continue   # skip corrupted/non-numeric filenames
            rows = ff.read_text(errors="ignore").split()
            if not rows:
                continue
            try:
                pts = np.array(rows, dtype=np.float32).reshape(-1, 3)  # (N, 3) = x,y,pid
            except ValueError:
                continue
            x_px  = pts[:, 0]
            y_px  = pts[:, 1]
            pids  = pts[:, 2].astype(int)
            xy_m  = _apply_homography(H, np.stack([x_px, y_px], axis=1))  # (N,2)
            frame_data[fid] = {int(p): xy_m[i] for i, p in enumerate(pids)}

        # ── Sliding window extraction ──────────────────────────────────────
        sorted_frames = sorted(frame_data.keys())
        T = len(sorted_frames)
        windows: list[np.ndarray] = []

        for start in range(0, T - self.seq_len + 1, _stride):
            wframes = sorted_frames[start: start + self.seq_len]
            present = set(frame_data[wframes[0]].keys())
            for f in wframes[1:]:
                present &= set(frame_data[f].keys())
            if len(present) < min_agents:
                continue
            peds = sorted(present)
            if max_agents is not None:
                peds = peds[:max_agents]
            N = len(peds)
            traj = np.zeros((N, self.seq_len, 2), dtype=np.float32)
            for t, f in enumerate(wframes):
                for n, pid in enumerate(peds):
                    traj[n, t] = frame_data[f][pid]
            windows.append(traj)

        # ── Split ──────────────────────────────────────────────────────────
        n     = len(windows)
        n_val = max(1, int(n * val_frac))
        n_tr  = n - n_val
        if split == "train":
            windows = windows[:n_tr]
        elif split == "val":
            windows = windows[n_tr:]

        self._windows  = windows
        self._M        = M
        self._C        = C
        assert windows, f"[GCS] No windows for split='{split}'"
        print(f"[GCS] split={split:5s}  windows={len(windows):5d}")
        disk_cache.save(windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        traj_np = self._windows[idx]
        tau0    = torch.from_numpy(traj_np).float()

        if self.max_agents is not None:
            N = tau0.shape[0]
            if N > self.max_agents:
                tau0 = tau0[: self.max_agents]
            elif N < self.max_agents:
                pad = torch.zeros(self.max_agents - N, self.seq_len, 2)
                tau0 = torch.cat([tau0, pad], dim=0)

        pos0 = tau0[:, 0, :]
        vel0 = (tau0[:, 1, :] - pos0) / self.dt
        goal = tau0[:, -1, :]
        S0   = torch.cat([pos0, vel0, goal], dim=-1)

        return tau0, self._M, self._C, S0
