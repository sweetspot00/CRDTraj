"""
Disk cache for processed trajectory dataset samples.

Usage
-----
    from data.cache import DatasetCache

    cache = DatasetCache(cache_dir="data/cache", name="ethucy_train_eth_8_12")

    if cache.exists():
        samples = cache.load()
    else:
        samples = expensive_preprocessing(...)
        cache.save(samples)

The cache file is a single .pt file saved via torch.save / torch.load.
The `name` should encode all parameters that affect the processed output
(split, seq_len, stride, max_agents, etc.) so that changing any parameter
automatically misses the cache.

Helpers
-------
cache_name(**kwargs) — build a canonical cache filename from keyword args.
"""

import hashlib
import torch
from pathlib import Path

# Central cache directory inside the CRDTraj project.
# All datasets default to saving their processed .pt files here so
# nothing is written into external data directories.
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / ".cache"


def cache_name(**kwargs) -> str:
    """
    Build a short, human-readable cache filename from keyword arguments.

    Parameters are sorted alphabetically so the name is deterministic.
    Values that are sets/lists are sorted before stringification.

    Example
    -------
    cache_name(dataset="ethucy", split="train", test_scene="eth",
               obs_len=8, pred_len=12, max_agents=8, stride=1)
    → "ethucy_train_eth_obs8_pred12_N8_stride1"
    """
    parts = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (set, list)):
            v = "_".join(str(x) for x in sorted(v))
        elif v is None:
            v = "none"
        parts.append(f"{k}{v}")
    name = "__".join(parts)
    # If the name is very long, append a short hash instead
    if len(name) > 120:
        h = hashlib.md5(name.encode()).hexdigest()[:12]
        name = name[:80] + "__" + h
    return name


class DatasetCache:
    """
    Simple torch.save / torch.load wrapper for processed dataset samples.

    Parameters
    ----------
    cache_dir : directory to store cache files (created if missing)
    name      : filename stem (without extension) — use cache_name() to build it
    """

    def __init__(self, cache_dir: str | Path, name: str):
        self._path = Path(cache_dir) / f"{name}.pt"

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def save(self, samples: object) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, str(self._path))
        print(f"[Cache] saved → {self._path}")

    def load(self) -> object:
        data = torch.load(str(self._path), map_location="cpu", weights_only=False)
        print(f"[Cache] loaded ← {self._path}")
        return data
