"""
Preprocessing script for CRDTraj.

Run this ONCE before training to process all raw datasets and save them
to CRDTraj/data/.cache/.  Training then loads instantly from cache.

Usage
-----
    python preprocess.py [options]

Examples
--------
    # Process everything (ETH/UCY + SDD + GCS + Synthetic)
    python preprocess.py \\
        --synth_results_root /path/to/sim/results \\
        --synth_json_root    /path/to/preprocessed_scene \\
        --synth_obstacle_root /path/to/pysfm_obstacles_meter_close_shape \\
        --sdd_root  /path/to/SDD \\
        --gcs_root  /path/to/GCS

    # ETH/UCY only
    python preprocess.py --datasets ethucy

    # Synthetic only, specific scenes and categories
    python preprocess.py \\
        --datasets synthetic \\
        --synth_results_root /path/to/sim/results \\
        --synth_json_root    /path/to/preprocessed_scene \\
        --synth_obstacle_root /path/to/pysfm_obstacles_meter_close_shape \\
        --synth_scene_ids "00_zurich,01_berkeley" \\
        --synth_categories "Escaping,Violent,Dense"
"""

import argparse
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Per-dataset preprocessors
# ---------------------------------------------------------------------------

def preprocess_ethucy(args) -> None:
    _section("ETH/UCY  (all 5 LOO splits × train/val/test)")
    import sys; sys.path.insert(0, ".")
    from data.eth_ucy import ETHUCYDataset

    scenes = ["eth", "hotel", "univ", "zara1", "zara2"]
    splits = ["train", "val", "test"]
    t_total = 0.0

    for test_scene in scenes:
        for split in splits:
            t0 = time.time()
            ds = ETHUCYDataset(
                root=args.ethucy_root,
                split=split,
                test_scene=test_scene,
                seq_len=args.seq_len,
                max_agents=args.max_agents,
                stride=args.stride,
            )
            elapsed = time.time() - t0
            t_total += elapsed
            print(f"  test={test_scene}  split={split:5s}  "
                  f"n={len(ds):5d}  ({elapsed:.1f}s)")

    print(f"\n  ETH/UCY done  total={_hms(t_total)}")


def preprocess_sdd(args) -> None:
    _section("SDD  (all scenes × train/val/test)")
    import sys; sys.path.insert(0, ".")
    from data.sdd import SDDDataset

    if not args.sdd_root:
        print("  skipped — pass --sdd_root to enable")
        return

    test_scenes_options = [
        None,                          # no LOO (all scenes in train/val)
        ["gates"],                     # common test split
    ]
    splits = ["train", "val"]
    t_total = 0.0

    # All scenes together (no LOO)
    for split in splits:
        t0 = time.time()
        ds = SDDDataset(
            sdd_root=args.sdd_root,
            split=split,
            seq_len=args.seq_len,
            max_agents=args.max_agents,
        )
        elapsed = time.time() - t0
        t_total += elapsed
        print(f"  split={split:5s}  n={len(ds):5d}  ({elapsed:.1f}s)")

    print(f"\n  SDD done  total={_hms(t_total)}")


def preprocess_gcs(args) -> None:
    _section("GCS  (train/val)")
    import sys; sys.path.insert(0, ".")
    from data.sdd import GCSDataset

    if not args.gcs_root:
        print("  skipped — pass --gcs_root to enable")
        return

    t_total = 0.0
    for split in ["train", "val"]:
        t0 = time.time()
        ds = GCSDataset(
            gcs_root=args.gcs_root,
            split=split,
            seq_len=args.seq_len,
            max_agents=args.max_agents,
        )
        elapsed = time.time() - t0
        t_total += elapsed
        print(f"  split={split:5s}  n={len(ds):5d}  ({elapsed:.1f}s)")

    print(f"\n  GCS done  total={_hms(t_total)}")


def preprocess_edin(args) -> None:
    _section("EDIN  (Edinburgh Informatics Forum, train/val/test)")
    import sys; sys.path.insert(0, ".")
    from data.edin import EDINDataset

    if not args.edin_root:
        print("  skipped — pass --edin_root to enable")
        return

    t_total = 0.0
    for split in ["train", "val", "test"]:
        t0 = time.time()
        ds = EDINDataset(
            edin_root=args.edin_root,
            split=split,
            seq_len=args.seq_len,
            max_agents=args.max_agents,
        )
        elapsed = time.time() - t0
        t_total += elapsed
        print(f"  split={split:5s}  n={len(ds):5d}  ({elapsed:.1f}s)")

    print(f"\n  EDIN done  total={_hms(t_total)}")


def preprocess_synthetic(args) -> None:
    _section("Synthetic SFM  (train/val)")
    import sys; sys.path.insert(0, ".")
    from data.synthetic import SyntheticDataset

    missing = [k for k in ("synth_results_root", "synth_json_root", "synth_obstacle_root")
               if not getattr(args, k)]
    if missing:
        print(f"  skipped — pass {', '.join('--'+m for m in missing)} to enable")
        return

    scene_ids = None
    if args.synth_scene_ids:
        scene_ids = [s.strip() for s in args.synth_scene_ids.split(",")]

    categories = None
    if args.synth_categories and args.synth_categories.upper() != "ALL":
        categories = [s.strip() for s in args.synth_categories.split(",")]

    t_total = 0.0
    for split in ["train", "val"]:
        t0 = time.time()
        ds = SyntheticDataset(
            results_root=args.synth_results_root,
            json_root=args.synth_json_root,
            obstacle_root=args.synth_obstacle_root,
            scene_ids=scene_ids,
            categories=categories,
            split=split,
            seq_len=args.seq_len,
            max_agents=args.max_agents,
        )
        elapsed = time.time() - t0
        t_total += elapsed
        print(f"  split={split:5s}  n={len(ds):6d}  ({elapsed:.1f}s)")

    print(f"\n  Synthetic done  total={_hms(t_total)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PREPROCESSORS = {
    "ethucy":    preprocess_ethucy,
    "sdd":       preprocess_sdd,
    "gcs":       preprocess_gcs,
    "edin":      preprocess_edin,
    "synthetic": preprocess_synthetic,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-process datasets and save to CRDTraj/data/.cache/"
    )

    # Which datasets
    p.add_argument(
        "--datasets", default="all",
        help="Comma-separated list of datasets to process: "
             "ethucy, sdd, gcs, edin, synthetic  (default: all)"
    )

    # Shared
    p.add_argument("--seq_len",    type=int, default=20)
    p.add_argument("--max_agents", type=int, default=8)
    p.add_argument("--stride",     type=int, default=1,
                   help="Window stride for ETH/UCY (default 1 = overlapping)")

    # ETH/UCY
    p.add_argument("--ethucy_root", default="data/eth_ucy")

    # SDD
    p.add_argument("--sdd_root", default=None,
                   help="Path to SDD root (annotation/, segmentation/, homography/)")

    # GCS
    p.add_argument("--gcs_root", default=None,
                   help="Path to GCS root (annotation/, homography/, segmentation/)")

    # EDIN
    p.add_argument("--edin_root", default=None,
                   help="Path to EDIN root (annotation/, homography/, segmentation/)")

    # Synthetic
    p.add_argument("--synth_results_root",  default=None)
    p.add_argument("--synth_json_root",     default=None)
    p.add_argument("--synth_obstacle_root", default=None)
    p.add_argument("--synth_scene_ids",     default=None,
                   help="Comma-separated short_names; default = all discovered scenes")
    p.add_argument("--synth_categories",    default=None,
                   help="Comma-separated categories; default = all non-Ambulatory; "
                        "pass ALL to include Ambulatory too")

    return p.parse_args()


def main():
    args = parse_args()

    if args.datasets.lower() == "all":
        selected = list(PREPROCESSORS.keys())
    else:
        selected = [d.strip().lower() for d in args.datasets.split(",")]

    from data.cache import DEFAULT_CACHE_DIR
    print(f"Cache directory: {DEFAULT_CACHE_DIR}")

    t_start = time.time()
    for name in selected:
        if name not in PREPROCESSORS:
            print(f"Unknown dataset '{name}', skipping.")
            continue
        PREPROCESSORS[name](args)

    print(f"\n{'='*60}")
    print(f"  All done  total wall time: {_hms(time.time() - t_start)}")
    print(f"  Cache: {DEFAULT_CACHE_DIR}")
    from data.cache import DEFAULT_CACHE_DIR as D
    pts = list(D.glob("*.pt"))
    total_mb = sum(f.stat().st_size for f in pts) / 1e6
    print(f"  Files: {len(pts)}  ({total_mb:.0f} MB total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
