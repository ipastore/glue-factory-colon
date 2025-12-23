# Quick script to check your dataset statistics
import argparse
import numpy as np
from pathlib import Path

DEFAULT_NPZ_DIR = Path(
    "/home/student/glue-factory-colon/data/Endomapper_CUDASIFT_DIC25/processed_npz"
)
DEFAULT_TRAIN_LIST = Path("gluefactory/datasets/endomapper_seq_lists/train_seqs.txt")
DEFAULT_VAL_LIST = Path("gluefactory/datasets/endomapper_seq_lists/val_seqs.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Endomapper map sizes.")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--train-list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--train-seqs", nargs="*", default=[])
    parser.add_argument("--val-seqs", nargs="*", default=[])
    return parser.parse_args()


args = parse_args()
npz_dir = args.npz_dir

print("Map Size Distribution:")
print("=" * 60)

sizes = []
split_hist = {"train": np.zeros(5, dtype=np.int64), "val": np.zeros(5, dtype=np.int64)}
split_images = {"train": 0, "val": 0}
train_seqs = set()
train_prefixes = []
val_seqs = set()
val_prefixes = []
split_enabled = False

if args.train_list and args.train_list.exists():
    split_enabled = True
    for line in args.train_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.count("_") <= 1:
            train_prefixes.append(line)
        else:
            train_seqs.add(line)

if args.val_list and args.val_list.exists():
    split_enabled = True
    for line in args.val_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.count("_") <= 1:
            val_prefixes.append(line)
        else:
            val_seqs.add(line)

if args.train_seqs:
    split_enabled = True
    for seq in args.train_seqs:
        if seq.count("_") <= 1:
            train_prefixes.append(seq)
        else:
            train_seqs.add(seq)

if args.val_seqs:
    split_enabled = True
    for seq in args.val_seqs:
        if seq.count("_") <= 1:
            val_prefixes.append(seq)
        else:
            val_seqs.add(seq)

for npz_file in sorted(npz_dir.glob("*.npz")):
    try:
        data = np.load(str(npz_file), allow_pickle=True)
        num_images = data["image_names"].shape[0]
        sizes.append((npz_file.stem, num_images))
        overlap_hist = ""
        overlap_counts = None
        if "overlap_matrix" in data:
            overlap_matrix = data["overlap_matrix"]
            if overlap_matrix.ndim == 2:
                triu = np.triu_indices_from(overlap_matrix, k=1)
                overlap_vals = overlap_matrix[triu]
            else:
                overlap_vals = overlap_matrix.ravel()
            masks = [
                (overlap_vals >= 0.0) & (overlap_vals < 0.1),
                (overlap_vals >= 0.1) & (overlap_vals < 0.3),
                (overlap_vals >= 0.3) & (overlap_vals < 0.5),
                (overlap_vals >= 0.5) & (overlap_vals < 0.7),
                (overlap_vals >= 0.7) & (overlap_vals <= 1.0),
            ]
            counts = [int(mask.sum()) for mask in masks]
            overlap_counts = counts
            overlap_hist = (
                " | ov 0.0-0.1:{:4d} 0.1-0.3:{:4d} 0.3-0.5:{:4d} "
                "0.5-0.7:{:4d} 0.7-1.0:{:4d}"
            ).format(*counts)
        if split_enabled and overlap_counts is not None:
            seq_name = data.get("seq", npz_file.stem.rsplit("_map", 1)[0])
            if isinstance(seq_name, np.ndarray):
                seq_name = seq_name.item()
            seq_name = str(seq_name)
            is_train = seq_name in train_seqs or any(
                seq_name.startswith(prefix) for prefix in train_prefixes
            )
            is_val = seq_name in val_seqs or any(
                seq_name.startswith(prefix) for prefix in val_prefixes
            )
            if is_train:
                split_hist["train"] += np.array(overlap_counts, dtype=np.int64)
                split_images["train"] += num_images
            if is_val:
                split_hist["val"] += np.array(overlap_counts, dtype=np.int64)
                split_images["val"] += num_images
        print(f"{npz_file.stem:30s}: {num_images:3d} images{overlap_hist}")
    except Exception as e:
        print(f"{npz_file.stem:30s}: ERROR - {e}")

if sizes:
    sizes_only = [s[1] for s in sizes]
    print("\n" + "=" * 60)
    print(f"Total maps: {len(sizes)}")
    print(f"Min images: {min(sizes_only)}")
    print(f"Max images: {max(sizes_only)}")
    print(f"Mean images: {np.mean(sizes_only):.1f}")
    print(f"Median images: {np.median(sizes_only):.1f}")

    if split_enabled:
        print("\n" + "=" * 60)
        print("Overlap Histogram Totals:")
        train_label = ",".join(train_prefixes + sorted(train_seqs))
        val_label = ",".join(val_prefixes + sorted(val_seqs))
        for split in ("train", "val"):
            counts = split_hist[split]
            total = int(counts.sum())
            if total > 0:
                perc = (counts / total) * 100.0
            else:
                perc = np.zeros_like(counts, dtype=np.float64)
            label = train_label if split == "train" else val_label
            print(
                "{}({}): total_images:{:4d} total_pairs:{:6d} "
                "ov 0.0-0.1:{:4d}({:5.1f}%) 0.1-0.3:{:4d}({:5.1f}%) "
                "0.3-0.5:{:4d}({:5.1f}%) 0.5-0.7:{:4d}({:5.1f}%) "
                "0.7-1.0:{:4d}({:5.1f}%)".format(
                    split,
                    label,
                    split_images[split],
                    total,
                    counts[0],
                    perc[0],
                    counts[1],
                    perc[1],
                    counts[2],
                    perc[2],
                    counts[3],
                    perc[3],
                    counts[4],
                    perc[4],
                )
            )
