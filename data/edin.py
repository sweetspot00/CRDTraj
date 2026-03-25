# obs_len / pred_len split is only for benchmark eval — training uses the full seq_len window
"""
Edinburgh Informatics Forum Pedestrian Dataset (EDIN).

Data source
-----------
Majecka, B. (2009). Statistical models of pedestrian behaviour in the Forum.
School of Informatics, University of Edinburgh.

Recorded overhead view of the Informatics Forum atrium at ~9 fps.
118 annotation files (one per recording day) from 2008–2010.

Layout expected
---------------
    edin_root/
        annotation/
            tracks.*.txt        (one file per day, MATLAB-style format)
        homography/
            edinburgh_H.txt     (3×3 pixel→world homography)
        segmentation/
            edinburgh_seg.png   (grayscale: 1=obstacle, 6=walkable)

Annotation format
-----------------
Each line of a tracks file contains one record:
    Properties.Rk=[...];  TRACK.Rk=[[x y frame];[x y frame];...];

where x, y are pixel coordinates and frame is an absolute frame index.
There is no agent ID across files — each file is independent.

Homography
----------
3×3 matrix H such that  [X, Y, W]^T = H @ [x_px, y_px, 1]^T.
World coordinates in metres: (X/W, Y/W).

Map
---
Grayscale segmentation image:
  - 6 = walkable (open floor / atrium)
  - 1 = obstacle  (walls, pillars, furniture)
Returns (3, 224, 224) float32 tensor.
"""

import hashlib
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.cache import DatasetCache, cache_name, DEFAULT_CACHE_DIR

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAP_SIZE   = 224
SEQ_LEN    = 20
DT         = 1.0 / 9.0     # ~9 fps raw; we subsample to ~2.5 fps
RAW_FPS    = 9.0
TARGET_FPS = 2.5

WALKABLE_LABEL = 6
OBSTACLE_LABEL = 1

EDIN_DESCRIPTION = (
    "Pedestrians walking through the Edinburgh Informatics Forum atrium, "
    "an indoor multi-level open space"
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TRACK_RE = re.compile(r"TRACK\.R\d+=(\[\[.+?\]\])", re.DOTALL)


def _parse_track_file(path: Path) -> list[np.ndarray]:
    """
    Parse one EDIN annotation file.

    Returns a list of arrays, each shape (T, 3): columns = [x_px, y_px, frame].
    """
    text = path.read_text(errors="replace")
    tracks: list[np.ndarray] = []

    for m in _TRACK_RE.finditer(text):
        body = m.group(1)  # "[[601 23 4471];[595 24 4472];...]"
        # Extract individual [x y frame] triplets
        pts = re.findall(r"\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]", body)
        if not pts:
            continue
        arr = np.array([[int(a), int(b), int(c)] for a, b, c in pts],
                       dtype=np.float32)   # (T, 3)
        tracks.append(arr)

    return tracks


def _load_tracks_per_file(annotation_dir: Path) -> list[list[np.ndarray]]:
    """
    Load all annotation files.

    Returns a list-of-lists: one inner list per recording day, each inner list
    contains the parsed tracks for that day.  Tracks from different days share
    independent frame-id spaces and must NOT be mixed into the same timeline.
    """
    result: list[list[np.ndarray]] = []
    for fpath in sorted(annotation_dir.glob("tracks.*.txt")):
        day_tracks = _parse_track_file(fpath)
        if day_tracks:
            result.append(day_tracks)
    return result


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------

def load_homography(h_path: Path) -> np.ndarray:
    """Load 3×3 homography matrix from text file."""
    H = np.loadtxt(str(h_path), dtype=np.float64)
    assert H.shape == (3, 3), f"Expected 3×3 homography, got {H.shape}"
    return H.astype(np.float32)


def pixel_to_world(pts_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography to pixel coordinates.

    pts_px : (N, 2)  [x_px, y_px]
    H      : (3, 3)

    Returns (N, 2) world coordinates in metres.
    """
    n = pts_px.shape[0]
    hom = np.ones((n, 3), dtype=np.float32)
    hom[:, :2] = pts_px
    w = (H @ hom.T).T          # (N, 3)
    return w[:, :2] / w[:, 2:3]


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def load_scene_map(seg_path: Path, size: int = MAP_SIZE) -> torch.Tensor:
    """
    Load EDIN segmentation map as (3, size, size) float32 tensor.

    Walkable (label 6) → 1.0
    Obstacle (label 1) → 0.0
    """
    assert _PIL, "Pillow required: pip install Pillow"
    img = Image.open(str(seg_path)).convert("L")
    arr = np.array(img, dtype=np.uint8)              # (H, W)
    walkable = (arr == WALKABLE_LABEL).astype(np.float32)

    t = torch.from_numpy(walkable).unsqueeze(0)      # (1, H, W)
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)                                     # (1, size, size)
    return t.expand(3, -1, -1).contiguous()          # (3, size, size)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def _subsample_stride(raw_fps: float, target_fps: float) -> int:
    return max(1, round(raw_fps / target_fps))


def _sort_and_resample(track: np.ndarray, stride: int) -> np.ndarray:
    """Sort by frame id, then subsample by stride. Returns (T, 2) in pixels."""
    idx  = np.argsort(track[:, 2])
    pts  = track[idx, :2]        # (T_raw, 2) pixel coords
    return pts[::stride]         # (T_sub, 2)


def _extract_windows_from_day(
    day_tracks: list[np.ndarray],
    H: np.ndarray,
    seq_len: int,
    stride: int,
    max_agents: int | None,
) -> list[np.ndarray]:
    """
    Extract sliding windows from one recording day's tracks.

    Converts pixel coords to world (metres), subsamples by stride, then
    slides a non-overlapping window of seq_len frames and collects all
    agents with complete data in that window.

    Returns list of (N_agents, seq_len, 2) float32 arrays.
    """
    # Convert tracks to world coords at target fps
    agent_seqs: list[tuple[np.ndarray, np.ndarray]] = []  # (frames_sub, world_xy)
    for track in day_tracks:
        sort_idx    = np.argsort(track[:, 2])
        frames_raw  = track[sort_idx, 2].astype(np.int64)
        pts_sorted  = track[sort_idx, :2]
        frames_sub  = frames_raw[::stride]
        pts_sub     = pts_sorted[::stride]
        if pts_sub.shape[0] < seq_len:
            continue
        world = pixel_to_world(pts_sub.astype(np.float32), H)  # (T_sub, 2)
        agent_seqs.append((frames_sub, world))

    if not agent_seqs:
        return []

    # Build a dense frame-index aligned to the union of all frames
    all_frames = sorted({int(f) for frames, _ in agent_seqs for f in frames})
    if len(all_frames) < seq_len:
        return []

    frame_to_gi: dict[int, int] = {f: i for i, f in enumerate(all_frames)}
    n_frames = len(all_frames)
    n_agents = len(agent_seqs)

    # Presence mask and world buffer
    present   = np.zeros((n_agents, n_frames), dtype=bool)
    world_buf = np.zeros((n_agents, n_frames, 2), dtype=np.float32)

    for ai, (frames, world) in enumerate(agent_seqs):
        for t_local, f in enumerate(frames):
            gi = frame_to_gi[int(f)]
            present[ai, gi]    = True
            world_buf[ai, gi] = world[t_local]

    # Sliding window over global frame timeline (non-overlapping)
    windows: list[np.ndarray] = []
    for start in range(0, n_frames - seq_len + 1, seq_len):
        cols  = slice(start, start + seq_len)
        valid = present[:, cols].all(axis=1)
        n_valid = int(valid.sum())
        if n_valid < 1:
            continue
        w = world_buf[valid][:, cols, :]          # (N, seq_len, 2)
        if max_agents is not None:
            w = w[:max_agents]
        windows.append(w.astype(np.float32))

    return windows


def _extract_windows(
    tracks_per_day: list[list[np.ndarray]],
    H: np.ndarray,
    seq_len: int,
    raw_fps: float,
    target_fps: float,
    max_agents: int | None,
) -> list[np.ndarray]:
    """Extract windows across all recording days."""
    stride  = _subsample_stride(raw_fps, target_fps)
    windows: list[np.ndarray] = []
    for day_tracks in tracks_per_day:
        windows.extend(
            _extract_windows_from_day(day_tracks, H, seq_len, stride, max_agents)
        )
    return windows


# ---------------------------------------------------------------------------
# Context embedding
# ---------------------------------------------------------------------------

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
        print("[EDIN] Warning: sentence-transformers not available; using zero context.")
        return None


# ---------------------------------------------------------------------------
# EDINDataset
# ---------------------------------------------------------------------------

class EDINDataset(Dataset):
    """
    Edinburgh Informatics Forum pedestrian dataset.

    Each sample returns (tau0, M, C, S0):
      tau0 : (N_agents, seq_len, 2)  world trajectory in metres  float32
      M    : (3, 224, 224)            scene walkability map        float32
      C    : (sent_dim,)              SBERT scene description      float32
      S0   : (N_agents, 6)            [x0,y0,vx0,vy0,gx,gy]       float32

    Parameters
    ----------
    edin_root   : directory with annotation/, homography/, segmentation/
    split       : 'train' | 'val' | 'test'
    seq_len     : trajectory window length in frames (at target_fps)
    max_agents  : cap on agents per window (default 8)
    val_frac    : fraction of data held out for validation (default 0.1)
    test_frac   : fraction held out for test (default 0.1)
    target_fps  : resample raw ~9 fps tracks to this fps (default 2.5)
    cache_dir   : disk cache location (default data/.cache/)
    """

    def __init__(
        self,
        edin_root: str,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = SEQ_LEN,
        max_agents: int | None = 8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        target_fps: float = TARGET_FPS,
        map_size: int = MAP_SIZE,
        sent_dim: int = 384,
        sbert_model: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.max_agents = max_agents
        self.target_dt  = 1.0 / target_fps

        edin_root  = Path(edin_root)
        cache_dir  = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        ckey  = cache_name(
            dataset="edin",
            split=split,
            seq=seq_len,
            agents=max_agents,
            fps=target_fps,
        )
        disk_cache = DatasetCache(cache_dir=cache_dir, name=ckey)

        # ── Load from cache if available ─────────────────────────────────
        if disk_cache.exists():
            data = disk_cache.load()
            if isinstance(data, tuple):
                self._windows, self._M, self._C = data
            else:
                self._windows = data
                self._M = load_scene_map(
                    edin_root / "segmentation" / "edinburgh_seg.png", map_size
                )
                sbert = _try_load_sbert(sbert_model)
                ctx_dir = cache_dir / "ctx"
                self._C = _embed(EDIN_DESCRIPTION, sent_dim, sbert, ctx_dir)
            print(f"[EDIN] split={split:5s}  windows={len(self._windows):5d}  (from cache)")
            return

        # ── Build from raw data ──────────────────────────────────────────
        ann_dir  = edin_root / "annotation"
        h_path   = edin_root / "homography" / "edinburgh_H.txt"
        seg_path = edin_root / "segmentation" / "edinburgh_seg.png"

        assert ann_dir.exists(),  f"[EDIN] annotation dir not found: {ann_dir}"
        assert h_path.exists(),   f"[EDIN] homography file not found: {h_path}"
        assert seg_path.exists(), f"[EDIN] segmentation file not found: {seg_path}"

        H         = load_homography(h_path)
        tracks_per_day = _load_tracks_per_file(ann_dir)

        windows = _extract_windows(
            tracks_per_day,
            H         = H,
            seq_len   = seq_len,
            raw_fps   = RAW_FPS,
            target_fps= target_fps,
            max_agents= max_agents,
        )

        # ── Train / val / test split ─────────────────────────────────────
        n        = len(windows)
        n_test   = max(1, int(n * test_frac))
        n_val    = max(1, int(n * val_frac))
        n_tr     = n - n_val - n_test
        if split == "train":
            windows = windows[:n_tr]
        elif split == "val":
            windows = windows[n_tr: n_tr + n_val]
        else:  # test
            windows = windows[n_tr + n_val:]

        assert windows, f"[EDIN] No windows for split='{split}'"

        M     = load_scene_map(seg_path, map_size)
        sbert = _try_load_sbert(sbert_model)
        ctx_dir = cache_dir / "ctx"
        C     = _embed(EDIN_DESCRIPTION, sent_dim, sbert, ctx_dir)

        self._windows = windows
        self._M       = M
        self._C       = C

        disk_cache.save((self._windows, self._M, self._C))
        print(f"[EDIN] split={split:5s}  windows={len(windows):5d}  agents≤{max_agents}")

    # ── Dataset protocol ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        traj_np = self._windows[idx]                          # (N, T, 2)
        tau0    = torch.from_numpy(traj_np).float()

        N = tau0.shape[0]
        if self.max_agents is not None and N < self.max_agents:
            pad  = torch.zeros(self.max_agents - N, self.seq_len, 2)
            tau0 = torch.cat([tau0, pad], dim=0)

        pos0 = tau0[:, 0, :]
        vel0 = (tau0[:, 1, :] - pos0) / self.target_dt
        goal = tau0[:, -1, :]
        S0   = torch.cat([pos0, vel0, goal], dim=-1)          # (N, 6)

        return tau0, self._M, self._C, S0
