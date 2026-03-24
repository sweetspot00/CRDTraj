"""
BalancedMixDataset — combine real (ETH/UCY) and synthetic datasets.

Each epoch sees exactly `total_size` samples drawn at a controlled real/synthetic
ratio.  The smaller dataset is cycled (wrapped) rather than truncated so no data
is silently discarded.

Every sample is normalised to the five-tuple format:
    (tau0, M, C, S0, meta)

ETH/UCY samples get a lightweight meta dict injected automatically:
    {"source": "real", "scene": scene_name}

Synthetic samples already carry a full meta dict; a "source" key is added.

Usage
-----
    from data import ETHUCYDataset
    from data.synthetic import SyntheticDataset
    from data.combined import BalancedMixDataset, mixed_collate_fn
    from torch.utils.data import DataLoader

    real_ds  = ETHUCYDataset(root="data/eth_ucy", split="train", test_scene="eth", max_agents=8)
    synth_ds = SyntheticDataset(sim_root=..., json_root=..., obstacle_root=..., max_agents=8)

    train_ds = BalancedMixDataset(real_ds, synth_ds, real_ratio=0.5)
    loader   = DataLoader(train_ds, batch_size=32, shuffle=True,
                          collate_fn=mixed_collate_fn, num_workers=4)

    tau0, M, C, S0, meta = next(iter(loader))
    # meta is a list of dicts, each with "source" in {"real", "synth"}
"""

import random
from typing import Literal

import torch
from torch.utils.data import Dataset


class BalancedMixDataset(Dataset):
    """
    Interleaves a real dataset and a synthetic dataset at a configurable ratio.

    Parameters
    ----------
    real_ds     : ETHUCYDataset (returns 4-tuple)
    synth_ds    : SyntheticDataset (returns 5-tuple with meta)
    real_ratio  : fraction of samples drawn from the real dataset (default 0.5)
    total_size  : total number of samples per epoch.
                  - None  → len(real_ds) + len(synth_ds)
                  - int   → explicit value (smaller dataset is cycled)
    shuffle     : if True, indices within each source are shuffled each epoch
                  (call .reshuffle() before every epoch, or set shuffle=True in
                  DataLoader which re-creates the index list each __getitem__ call)
    seed        : random seed for reproducible shuffles
    """

    def __init__(
        self,
        real_ds: Dataset,
        synth_ds: Dataset,
        real_ratio: float = 0.5,
        total_size: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        assert 0.0 < real_ratio < 1.0, "real_ratio must be in (0, 1)"
        self.real_ds    = real_ds
        self.synth_ds   = synth_ds
        self.real_ratio = real_ratio
        self.shuffle    = shuffle
        self._rng       = random.Random(seed)

        n_real_total  = len(real_ds)
        n_synth_total = len(synth_ds)

        if total_size is None:
            total_size = n_real_total + n_synth_total

        self._total   = total_size
        self._n_real  = round(total_size * real_ratio)
        self._n_synth = total_size - self._n_real

        self._build_index()

        print(
            f"[BalancedMix] real={n_real_total} synth={n_synth_total} "
            f"→ epoch size={self._total} "
            f"(real={self._n_real} [{real_ratio*100:.0f}%], "
            f"synth={self._n_synth} [{(1-real_ratio)*100:.0f}%])"
        )

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """
        Build a flat index list of (source, dataset_index) pairs.

        Cycling: if the requested count exceeds the dataset size, indices wrap
        around modulo len(dataset).  This is equivalent to repeating the dataset
        until the quota is met — no sample is dropped.
        """
        n_real  = len(self.real_ds)
        n_synth = len(self.synth_ds)

        real_pool  = list(range(n_real))
        synth_pool = list(range(n_synth))

        if self.shuffle:
            self._rng.shuffle(real_pool)
            self._rng.shuffle(synth_pool)

        # Tile the pools to cover the requested counts
        real_idx  = [real_pool[i % n_real]   for i in range(self._n_real)]
        synth_idx = [synth_pool[i % n_synth] for i in range(self._n_synth)]

        # Interleave: pair each real sample with a synth sample where possible
        combined = []
        ri, si = iter(real_idx), iter(synth_idx)
        for r, s in zip(ri, si):
            combined.append(("real",  r))
            combined.append(("synth", s))
        # Append any remainder (when counts differ)
        for r in ri:
            combined.append(("real",  r))
        for s in si:
            combined.append(("synth", s))

        if self.shuffle:
            self._rng.shuffle(combined)

        self._index: list[tuple[Literal["real", "synth"], int]] = combined

    def reshuffle(self) -> None:
        """Call at the start of each epoch to re-randomise the index."""
        self._build_index()

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        source, ds_idx = self._index[idx]

        if source == "real":
            tau0, M, C, S0 = self.real_ds[ds_idx]
            # Inject minimal meta so callers always get a 5-tuple
            meta = {"source": "real"}
        else:
            tau0, M, C, S0, meta = self.synth_ds[ds_idx]
            meta = {**meta, "source": "synth"}

        return tau0, M, C, S0, meta

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def real_weight(self) -> float:
        return self._n_real / self._total

    @property
    def synth_weight(self) -> float:
        return self._n_synth / self._total


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def mixed_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]]
):
    """
    Collate a batch of mixed (real + synthetic) samples.

    Handles variable agent counts by padding to N_max within the batch.

    Returns
    -------
    tau0 : (B, N_max, T, 2)
    M    : (B, 3, H, W)
    C    : (B, sent_dim)
    S0   : (B, N_max, 6)
    meta : list[dict]   length B, each dict has "source" key
    """
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
