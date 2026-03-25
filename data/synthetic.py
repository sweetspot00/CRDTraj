# obs_len / pred_len split is only for benchmark eval — training uses the full seq_len window
"""
Synthetic dataset loader for CRDTraj.

Data produced by a Social Force Model (pySFM) driven by LLM-generated behavioural
specifications.

Top-level directory layout
---------------------------
  results_root/
    {short_name}/             e.g. 00_zurich, 01_berkeley, ...
      simulations/            sim_{scene_id}_{timestamp}.npz
      configs/                {NNNN}_{scene_id}_{model}_{timestamp}.toml

  json_root/
    {short_name}/             {NNNN}_{scene_id}.json

  obstacle_root/
    {scene_id}_anchored.npz  (scene_id extracted from sim filenames)

Passing scene_ids=None auto-discovers all valid scenes under results_root.
Passing a list of short_name strings (e.g. ["00_zurich", "01_berkeley"]) restricts
the dataset to those scenes.

NPZ  sim format
---------------
  states : (T_steps, N_agents, 7)
           cols: x_m, y_m, vx, vy, goal_x, goal_y, weight
  scene  : 0-d object → Python dict with scenario metadata

Obstacle NPZ format
-------------------
  obstacles : (N_segs, 4)  each row = (x0, x1, y0, y1) in metres

TOML config
-----------
  scene.step_width  — simulation time step Δt (seconds)

JSON scenario
-------------
  scenario, category, towards_event, event_center_m, goals_m, initial_state, groups

Returns (tau0, M, C, S0) in the same format as ETHUCYDataset plus a metadata dict.
"""

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Literal

from data.cache import DatasetCache, cache_name, DEFAULT_CACHE_DIR

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.ndimage import binary_dilation
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAP_SIZE = 224
SEQ_LEN_DEFAULT = 20
TARGET_DT_DEFAULT = 0.4   # seconds

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Ambulatory":    "pedestrians casually walking through the area",
    "Cohesive":      "a group converging on a common focal point together",
    "Escaping":      "crowd escaping rapidly from a perceived threat",
    "Violent":       "violent altercation causing people to flee in panic",
    "Expressive":    "expressive crowd gathering around a focal event",
    "Aggressive":    "aggressive confrontation drawing onlookers and participants",
    "Rushing":       "crowd rushing toward entry or exit points",
    "Dense":         "dense crowd moving through a congested area",
    "Disability":    "mixed crowd including people with mobility impairments",
    "Demonstrator":  "protest demonstration crowd moving with purpose",
    "Participatory": "crowd actively participating in a shared event",
}

# Categories well-covered by ETH/UCY real data — excluded from synthetic by default
# so the synthetic data complements rather than duplicates the real data.
AMBULATORY_CATEGORIES = {"Ambulatory"}

# All non-ambulatory crowd categories
CROWD_CATEGORIES = set(CATEGORY_DESCRIPTIONS.keys()) - AMBULATORY_CATEGORIES

# Short directory names to skip when auto-discovering scenes
_SKIP_DIRS = {"configs", "ignore", "metrics", "simulations", "analytics", "cache"}


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _extract_scene_id(sim_path: Path) -> str:
    """
    Extract scene_id from a sim filename.

    Pattern:  sim_{scene_id}_{13-digit-timestamp}.npz
    The timestamp is a long integer at the end; everything in between is the scene_id.
    """
    name = sim_path.stem          # e.g. "sim_00_Zurich_HB_simplified_obstacle_1767695456"
    name = re.sub(r"^sim_", "", name)
    name = re.sub(r"_\d{10,}$", "", name)   # strip trailing Unix timestamp
    return name


def _find_obstacle_npz(obstacle_dir: Path, scene_id: str) -> Path | None:
    """Try {scene_id}_anchored.npz first, then {scene_id}.npz."""
    for pattern in (f"{scene_id}_anchored.npz", f"{scene_id}.npz"):
        matches = sorted(obstacle_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_scenes(
    results_root: Path,
    json_root: Path,
    obstacle_root: Path,
    include: list[str] | None = None,   # short_name whitelist; None = all
) -> list[dict]:
    """
    Return a list of scene dicts, one per valid scene directory:

        {
          "short_name":  "00_zurich",
          "scene_id":    "00_Zurich_HB_simplified_obstacle",
          "sim_dir":     Path(...)/00_zurich/simulations,
          "cfg_dir":     Path(...)/00_zurich/configs,
          "json_dir":    Path(...)/json_root/00_zurich,
          "obstacle":    Path(...)/obstacle_root/...npz,
        }

    A scene is valid if it has ≥ 1 sim NPZ, a matching obstacle file, and a JSON dir.
    Duplicate scene_ids (e.g. 22_hong / 22_hong_kong_airport) keep only the first.
    """
    seen_scene_ids: set[str] = set()
    scenes: list[dict] = []

    for scene_dir in sorted(results_root.iterdir()):
        short = scene_dir.name
        if not scene_dir.is_dir() or short in _SKIP_DIRS:
            continue
        if include is not None and short not in include:
            continue

        sim_dir = scene_dir / "simulations"
        cfg_dir = scene_dir / "configs"
        sim_files = sorted(sim_dir.glob("sim_*.npz")) if sim_dir.exists() else []
        if not sim_files:
            continue

        scene_id = _extract_scene_id(sim_files[0])
        if scene_id in seen_scene_ids:
            continue   # skip duplicate

        obs_path = _find_obstacle_npz(obstacle_root, scene_id)
        if obs_path is None:
            continue

        json_dir = json_root / short
        if not json_dir.exists():
            continue

        seen_scene_ids.add(scene_id)
        scenes.append({
            "short_name": short,
            "scene_id":   scene_id,
            "sim_dir":    sim_dir,
            "cfg_dir":    cfg_dir,
            "json_dir":   json_dir,
            "obstacle":   obs_path,
        })

    return scenes


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _find_config(cfg_dir: Path, scene_index: int) -> Path | None:
    matches = sorted(cfg_dir.glob(f"{scene_index:04d}_*.toml"))
    return matches[0] if matches else None


def _find_json(json_dir: Path, scene_index: int) -> Path | None:
    matches = sorted(json_dir.glob(f"{scene_index:04d}_*.json"))
    return matches[0] if matches else None


def _load_sim(path: Path) -> tuple[np.ndarray, dict]:
    npz = np.load(path, allow_pickle=True)
    states = npz["states"].astype(np.float32)
    scene  = npz["scene"].item()
    return states, scene


def _load_config_dt(toml_path: Path) -> float:
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    return float(cfg["scene"]["step_width"])


def _load_json(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Trajectory subsampling & window extraction
# ---------------------------------------------------------------------------

def subsample_states(states: np.ndarray, sim_dt: float, target_dt: float) -> np.ndarray:
    stride = max(1, round(target_dt / sim_dt))
    return states[::stride]


def extract_windows(
    states: np.ndarray,
    seq_len: int,
    stride: int = 1,
    min_agents: int = 2,
    max_agents: int | None = None,
) -> list[np.ndarray]:
    """Return list of (N, seq_len, 2) position arrays."""
    T, N, _ = states.shape
    if T < seq_len or N < min_agents:
        return []
    if max_agents is not None:
        N = min(N, max_agents)
    windows = []
    for start in range(0, T - seq_len + 1, stride):
        traj = states[start : start + seq_len, :N, :2]
        windows.append(traj.transpose(1, 0, 2).copy())
    return windows


# ---------------------------------------------------------------------------
# Obstacle map rasterisation
# ---------------------------------------------------------------------------

def rasterize_obstacles(
    segments: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    size: int = MAP_SIZE,
    dilate_px: int = 2,
) -> torch.Tensor:
    """
    Rasterise (N_seg, 4) obstacle segments = (x0, x1, y0, y1) metres onto a
    (3, size, size) binary map. 1 = wall, 0 = free.
    """
    grid = np.zeros((size, size), dtype=np.float32)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    def _to_px(x, y):
        px = np.clip((x - x_min) / x_range * (size - 1), 0, size - 1).astype(np.int32)
        py = np.clip((y - y_min) / y_range * (size - 1), 0, size - 1).astype(np.int32)
        return px, py

    for x0, x1, y0, y1 in segments:
        seg_len_px = math.sqrt(
            ((x1 - x0) / x_range * (size - 1)) ** 2 +
            ((y1 - y0) / y_range * (size - 1)) ** 2
        )
        n_steps = max(2, int(math.ceil(seg_len_px)) + 1)
        xs = np.linspace(x0, x1, n_steps)
        ys = np.linspace(y0, y1, n_steps)
        pxs, pys = _to_px(xs, ys)
        grid[pys, pxs] = 1.0

    if dilate_px > 0:
        if _SCIPY:
            y_d, x_d = np.ogrid[-dilate_px : dilate_px + 1, -dilate_px : dilate_px + 1]
            disk = (x_d ** 2 + y_d ** 2) <= dilate_px ** 2
            grid = binary_dilation(grid > 0, structure=disk).astype(np.float32)
        else:
            import torch.nn.functional as F
            t = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
            t = F.max_pool2d(t, kernel_size=2 * dilate_px + 1, stride=1, padding=dilate_px)
            grid = t.squeeze().numpy()

    map_t = torch.from_numpy(grid.clip(0, 1))
    return map_t.unsqueeze(0).expand(3, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# Context embedding
# ---------------------------------------------------------------------------

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
        print("[Synthetic] Warning: sentence-transformers not available; using zero context.")
        return None


# ---------------------------------------------------------------------------
# SyntheticDataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Multi-scene synthetic dataset (SFM-simulated, LLM-configured).

    Automatically discovers all valid scenes under results_root, or restricts
    to the short_names listed in scene_ids.

    Each sample returns (tau0, M, C, S0, meta):
      tau0 : (N, T, 2)        trajectory in metres          float32
      M    : (3, 224, 224)    binary obstacle map [0=free]  float32
      C    : (sent_dim,)      SBERT context embedding       float32
      S0   : (N, 6)           [x0,y0,vx0,vy0,gx,gy]        float32
      meta : dict             scene_index, scenario, category, source info

    Parameters
    ----------
    results_root  : top-level sim/results/ directory
    json_root     : top-level preprocessed_scene/ directory
    obstacle_root : directory containing *_anchored.npz files
    scene_ids     : list of short_name strings to include (None = all valid scenes)
    categories    : set/list of category strings to include.
                    Defaults to CROWD_CATEGORIES (all except "Ambulatory"), since
                    ETH/UCY already covers normal walking.  Pass None to keep all.
    split         : 'train' | 'val' | 'test'
    seq_len       : total trajectory window length in frames
    target_dt     : resample all sims to this Δt (seconds)
    min_agents    : skip windows with fewer agents
    max_agents    : pad/truncate to this agent count (None = variable)
    map_size      : obstacle map side (pixels)
    dilate_px     : obstacle line dilation (pixels)
    sent_dim      : SBERT output dimension
    sbert_model   : SentenceTransformer model name
    val_frac      : fraction of windows held out for validation
    stride        : window extraction stride (default = seq_len → non-overlapping)
    """

    def __init__(
        self,
        results_root: str,
        json_root: str,
        obstacle_root: str,
        scene_ids: list[str] | None = None,
        categories: set[str] | list[str] | None = CROWD_CATEGORIES,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = SEQ_LEN_DEFAULT,
        target_dt: float = TARGET_DT_DEFAULT,
        min_agents: int = 2,
        max_agents: int | None = None,
        map_size: int = MAP_SIZE,
        dilate_px: int = 2,
        sent_dim: int = 384,
        sbert_model: str = "all-MiniLM-L6-v2",
        val_frac: float = 0.1,
        stride: int | None = None,
        cache_dir: str | None = None,   # set to save/load processed samples
    ):
        super().__init__()
        self.seq_len      = seq_len
        self.target_dt    = target_dt
        self.max_agents   = max_agents
        self.map_size     = map_size
        self.sent_dim     = sent_dim
        self._categories  = set(categories) if categories is not None else None

        results_root  = Path(results_root)
        json_root     = Path(json_root)
        obstacle_root = Path(obstacle_root)
        _stride = stride if stride is not None else self.seq_len

        # ── Disk cache for processed samples ──────────────────────────────
        _cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        _cname = cache_name(
            dataset="synthetic",
            split=split,
            scenes=sorted(scene_ids) if scene_ids else "all",
            cats=sorted(categories) if categories else "all",
            seq=seq_len,
            dt=target_dt, stride=_stride,
            N=max_agents or "var",
            dilate=dilate_px,
            valf=val_frac,
        )
        disk_cache = DatasetCache(_cache_dir, _cname)

        if disk_cache.exists():
            self._samples = disk_cache.load()
            cats_str = ", ".join(sorted(self._categories)) if self._categories else "ALL"
            print(
                f"[Synthetic] loaded from cache  split={split:5s}  "
                f"windows={len(self._samples):6d}  categories=[{cats_str}]"
            )
            return

        # ── Discover scenes ────────────────────────────────────────────────
        scenes = discover_scenes(results_root, json_root, obstacle_root, include=scene_ids)
        assert scenes, f"No valid scenes found under {results_root}"

        # ── SBERT + ctx cache ──────────────────────────────────────────────
        sbert     = _try_load_sbert(sbert_model)
        ctx_cache = results_root / "cache" / "ctx"

        # ── Load all samples ───────────────────────────────────────────────
        self._samples: list[tuple[np.ndarray, torch.Tensor, torch.Tensor, dict]] = []

        scene_window_counts = []

        for sc in scenes:
            short    = sc["short_name"]
            scene_id = sc["scene_id"]
            sim_dir  = sc["sim_dir"]
            cfg_dir  = sc["cfg_dir"]
            json_dir = sc["json_dir"]

            # Build obstacle map once per scene
            segs = np.load(sc["obstacle"])["obstacles"]
            x_min = float(segs[:, :2].min())
            x_max = float(segs[:, :2].max())
            y_min = float(segs[:, 2:].min())
            y_max = float(segs[:, 2:].max())
            obstacle_map = rasterize_obstacles(
                segs, x_min, x_max, y_min, y_max,
                size=map_size, dilate_px=dilate_px,
            )

            n_before = len(self._samples)

            for sim_path in sorted(sim_dir.glob("sim_*.npz")):
                states, scene_info = _load_sim(sim_path)
                scene_index = scene_info["scene_index"]

                cfg_path = _find_config(cfg_dir, scene_index)
                sim_dt   = _load_config_dt(cfg_path) if cfg_path else target_dt

                json_path = _find_json(json_dir, scene_index)
                jdata     = _load_json(json_path) if json_path else scene_info
                scenario  = jdata.get("scenario", scene_info.get("scenario", ""))
                category  = jdata.get("category", "")

                # ── Category filter ────────────────────────────────────────
                if self._categories is not None and category not in self._categories:
                    continue

                ctx_text = f"{scenario} [{CATEGORY_DESCRIPTIONS.get(category, category)}]"
                C = _embed(ctx_text, sent_dim, sbert, ctx_cache)

                meta = {
                    "scene_index":    scene_index,
                    "short_name":     short,
                    "scene_id":       scene_id,
                    "scenario":       scenario,
                    "category":       category,
                    "towards_event":  jdata.get("towards_event", False),
                    "event_center_m": jdata.get("event_center_m"),
                    "goals_m":        jdata.get("goals_m"),
                    "dt":             sim_dt,
                    "sim_path":       str(sim_path),
                }

                sub = subsample_states(states, sim_dt, target_dt)
                windows = extract_windows(
                    sub,
                    seq_len=self.seq_len,
                    stride=_stride,
                    min_agents=min_agents,
                    max_agents=max_agents,
                )
                for w in windows:
                    self._samples.append((w, obstacle_map, C, meta))

            n_added = len(self._samples) - n_before
            scene_window_counts.append((short, n_added))

        # ── Train / val / test split ───────────────────────────────────────
        n     = len(self._samples)
        n_val = max(1, int(n * val_frac))
        n_tr  = n - n_val
        if split == "train":
            self._samples = self._samples[:n_tr]
        elif split == "val":
            self._samples = self._samples[n_tr:]
        # test → keep all

        assert self._samples, f"No windows found for split='{split}'"

        cats_str = ", ".join(sorted(self._categories)) if self._categories else "ALL"
        print(
            f"[Synthetic] {len(scenes)} scenes  split={split:5s}  "
            f"windows={len(self._samples):6d}  "
            f"(total before split: {n})  "
            f"categories=[{cats_str}]"
        )
        disk_cache.save(self._samples)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        traj_np, M, C, meta = self._samples[idx]

        tau0 = torch.from_numpy(traj_np).float()

        if self.max_agents is not None:
            N = tau0.shape[0]
            if N > self.max_agents:
                tau0 = tau0[: self.max_agents]
            elif N < self.max_agents:
                pad = torch.zeros(self.max_agents - N, self.seq_len, 2)
                tau0 = torch.cat([tau0, pad], dim=0)

        S0 = self._compute_s0(tau0, meta)
        return tau0, M, C, S0, meta

    def _compute_s0(self, tau0: torch.Tensor, meta: dict) -> torch.Tensor:
        pos0 = tau0[:, 0, :]
        vel0 = (tau0[:, 1, :] - pos0) / self.target_dt
        goal = tau0[:, -1, :]
        return torch.cat([pos0, vel0, goal], dim=-1)

    @property
    def n_scenes(self) -> int:
        return len({m["scene_id"] for _, _, _, m in self._samples})


# ---------------------------------------------------------------------------
# Collate fn
# ---------------------------------------------------------------------------

def synthetic_collate_fn(batch):
    """Collate variable-agent samples. Returns (tau0, M, C, S0, meta_list)."""
    tau0_list, M_list, C_list, S0_list, meta_list = zip(*batch)

    N_max = max(t.shape[0] for t in tau0_list)
    T     = tau0_list[0].shape[1]
    B     = len(batch)

    tau0_pad = torch.zeros(B, N_max, T, 2)
    S0_pad   = torch.zeros(B, N_max, 6)

    for i, (tau0, s0) in enumerate(zip(tau0_list, S0_list)):
        N = tau0.shape[0]
        tau0_pad[i, :N] = tau0
        S0_pad[i,   :N] = s0

    return (
        tau0_pad,
        torch.stack(M_list),
        torch.stack(C_list),
        S0_pad,
        list(meta_list),
    )
