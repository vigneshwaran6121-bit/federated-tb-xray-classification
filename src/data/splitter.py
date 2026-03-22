"""
splitter.py
Splits the raw dataset into:
  - 3 hospital node splits (federated training)
  - 1 global held-out test set
Supports both IID and non-IID partitioning.
"""

import os
import shutil
import random
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Load config ─────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ── Gather all image paths per class ────────────────────────
def gather_images(raw_dir: str, classes: list) -> dict:
    """Returns dict: {class_name: [list of image paths]}"""
    image_map = defaultdict(list)
    for cls in classes:
        cls_dir = Path(raw_dir) / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")
        images = list(cls_dir.glob("*.png")) + \
                 list(cls_dir.glob("*.jpg")) + \
                 list(cls_dir.glob("*.jpeg"))
        image_map[cls] = [str(p) for p in images]
        print(f"  [{cls}] Found {len(images)} images")
    return image_map

# ── Create global test set ───────────────────────────────────
def split_test_set(image_map: dict, test_split: float, seed: int) -> tuple:
    """
    Carves out a global test set from ALL classes.
    Returns: (remaining_map, test_map)
    """
    random.seed(seed)
    remaining = {}
    test_set  = defaultdict(list)

    for cls, paths in image_map.items():
        shuffled = paths[:]
        random.shuffle(shuffled)
        n_test = int(len(shuffled) * test_split)
        test_set[cls]  = shuffled[:n_test]
        remaining[cls] = shuffled[n_test:]
        print(f"  [{cls}] Test: {n_test} | Remaining: {len(remaining[cls])}")

    return remaining, test_set

# ── IID split across N nodes ─────────────────────────────────
def iid_split(remaining_map: dict, num_clients: int, seed: int) -> list:
    """
    Splits data equally and randomly across nodes.
    Each node gets a representative sample of all classes.
    Returns: list of dicts [{class: [paths]}, ...]
    """
    random.seed(seed)
    node_data = [defaultdict(list) for _ in range(num_clients)]

    for cls, paths in remaining_map.items():
        shuffled = paths[:]
        random.shuffle(shuffled)
        chunks = np.array_split(shuffled, num_clients)
        for i, chunk in enumerate(chunks):
            node_data[i][cls] = list(chunk)

    return node_data

# ── Non-IID split across N nodes ────────────────────────────
def non_iid_split(remaining_map: dict, num_clients: int, seed: int) -> list:
    """
    Dirichlet-based non-IID split.
    Simulates real-world hospital imbalance (e.g., one hospital
    sees more TB cases than others).
    alpha=0.5 — moderately heterogeneous.
    """
    random.seed(seed)
    np.random.seed(seed)
    node_data = [defaultdict(list) for _ in range(num_clients)]
    alpha = 0.5  # lower = more heterogeneous

    for cls, paths in remaining_map.items():
        shuffled = paths[:]
        random.shuffle(shuffled)
        # Dirichlet proportions
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(shuffled)).astype(int)
        # Fix rounding so we don't lose images
        proportions[-1] = len(shuffled) - proportions[:-1].sum()

        idx = 0
        for i, count in enumerate(proportions):
            node_data[i][cls] = shuffled[idx: idx + count]
            idx += count

    return node_data

# ── Copy files into destination folders ─────────────────────
def copy_split(split_data: dict, dest_dir: Path):
    """Copies images to dest_dir/ClassName/image.png"""
    for cls, paths in split_data.items():
        cls_dir = dest_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy2(src, cls_dir / Path(src).name)

# ── Print split summary ──────────────────────────────────────
def print_summary(node_splits: list, test_map: dict, classes: list):
    print("\n" + "="*55)
    print("         DATASET SPLIT SUMMARY")
    print("="*55)
    for i, node in enumerate(node_splits):
        total = sum(len(v) for v in node.values())
        print(f"\n  Hospital Node {i+1}  (total: {total} images)")
        for cls in classes:
            print(f"    {cls:15s}: {len(node.get(cls, []))} images")
    print(f"\n  Global Test Set")
    for cls in classes:
        print(f"    {cls:15s}: {len(test_map.get(cls, []))} images")
    print("="*55 + "\n")

# ── Main ─────────────────────────────────────────────────────
def main(split_type: str = "iid"):
    cfg      = load_config()
    seed     = cfg["project"]["seed"]
    classes  = cfg["data"]["classes"]
    raw_dir  = cfg["data"]["raw_dir"]
    test_dir = cfg["data"]["test_dir"]
    splits_dir   = cfg["data"]["splits_dir"]
    test_split   = cfg["data"]["test_split"]
    num_clients  = cfg["federated"]["num_clients"]

    print(f"\n{'='*55}")
    print(f"  Federated Data Splitter — mode: {split_type.upper()}")
    print(f"{'='*55}\n")

    # 1. Gather images
    print("► Gathering images...")
    image_map = gather_images(raw_dir, classes)

    # 2. Carve out test set
    print("\n► Creating global test set...")
    remaining_map, test_map = split_test_set(image_map, test_split, seed)

    # 3. Split among nodes
    print(f"\n► Splitting into {num_clients} hospital nodes ({split_type})...")
    if split_type == "iid":
        node_splits = iid_split(remaining_map, num_clients, seed)
    else:
        node_splits = non_iid_split(remaining_map, num_clients, seed)

    # 4. Print summary
    print_summary(node_splits, test_map, classes)

    # 5. Copy files
    print("► Copying files to split directories...")

    # Test set
    copy_split(test_map, Path(test_dir))
    print(f"  ✓ Test set → {test_dir}/")

    # Node splits
    for i, node_data in enumerate(node_splits):
        node_dir = Path(splits_dir) / f"node_{i+1}"
        copy_split(node_data, node_dir)
        print(f"  ✓ Node {i+1}  → {node_dir}/")

    print("\n✅ Data split complete!\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_type",
        type=str,
        default="iid",
        choices=["iid", "non_iid"],
        help="IID = equal distribution | non_iid = hospital imbalance simulation"
    )
    args = parser.parse_args()
    main(split_type=args.split_type)