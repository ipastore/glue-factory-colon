"""Inspect HDF5 feature files exported by glue-factory."""
import argparse
from pathlib import Path

import h5py
import numpy as np


def print_dataset_info(name: str, obj: h5py.Dataset) -> None:
    """Print information about an HDF5 dataset."""
    print(f"  {name}:")
    print(f"    shape: {obj.shape}")
    print(f"    dtype: {obj.dtype}")
    # Only print values for very small datasets
    if obj.size > 0 and obj.size < 10:
        print(f"    value: {obj[...]}")


def inspect_h5_file(filepath: Path, show_samples: bool = False) -> None:
    """Inspect an HDF5 file and print its structure."""
    print(f"\n{'='*80}")
    print(f"File: {filepath.name}")
    print(f"Size: {filepath.stat().st_size / 1024**2:.2f} MB")
    print(f"{'='*80}")

    with h5py.File(filepath, "r") as f:
        print(f"\nTop-level keys: {list(f.keys())}")

        for key in f.keys():
            print(f"\n[{key}]")
            group = f[key]

            if isinstance(group, h5py.Group):
                print(f"  Type: Group")
                print(f"  Keys: {list(group.keys())}")

                # Print info for each dataset in the group
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    if isinstance(dataset, h5py.Dataset):
                        print_dataset_info(dataset_name, dataset)

                # Show a sample if requested (read only first 5 rows, don't load all)
                if show_samples and "keypoints" in group:
                    kpts = group["keypoints"]
                    print(f"\n  Sample keypoints (first 5):")
                    # Read only the first 5 items without loading the entire array
                    sample_size = min(5, kpts.shape[0])
                    print(f"    {kpts[:sample_size]}")

            elif isinstance(group, h5py.Dataset):
                print_dataset_info("", group)

    print()  # Add newline at the end


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 feature files from glue-factory"
    )
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Path(s) to .h5 file(s) to inspect",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Show sample data from datasets",
    )
    args = parser.parse_args()

    for filepath_str in args.files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue

        try:
            inspect_h5_file(filepath, show_samples=args.samples)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


if __name__ == "__main__":
    main()