"""Check for corrupted images in the dataset."""
import traceback
import argparse
from pathlib import Path
from tqdm import tqdm

# Try to import OpenCV; optional but useful
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

def check_image(path: Path, black_threshold: int = 0):
    """Check if an image decodes and is not fully black."""
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"

    if cv2 is None:
        return False, "OpenCV (cv2) not available in this environment"

    try:
        # mirror read_image: read in color by default
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return False, "cv2.imread returned None (could not decode image)"
        if np is not None:
            max_value = int(image.max())
            if max_value <= black_threshold:
                return (
                    False,
                    f"image is fully black (max_pixel={max_value}, threshold={black_threshold})",
                )
        return True, None
    except Exception as e:
        return False, f"cv2.imread exception: {e}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check images that cannot be decoded or are fully black."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/media/student/HDD/nacho/glue-factory/data/slam-results_long_sequences_ENE26/"),
        help="Root directory to scan for images.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split subfolders to scan under data_dir. If omitted, scan data_dir recursively.",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=0,
        help="Treat an image as black if all pixel values are <= this threshold.",
    )
    return parser.parse_args()


def find_images(root: Path):
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    images = []
    for pattern in patterns:
        images.extend(root.rglob(pattern))
    return sorted(set(images))


def run_check(scan_dir: Path, label: str, output_path: Path, black_threshold: int):
    print(f"\nChecking {label}...")
    images = find_images(scan_dir)
    print(f"Found {len(images)} images")

    corrupted = []
    for img_path in tqdm(images):
        try:
            ok, error = check_image(img_path, black_threshold=black_threshold)
        except Exception:
            ok = False
            error = f"Unhandled exception:\n{traceback.format_exc()}"
        if not ok:
            corrupted.append((img_path, error))
            print(f"Corrupted: {img_path} - {error}")

    print(f"\nFound {len(corrupted)} corrupted/black images in {label}")

    if corrupted:
        with open(output_path, "w") as f:
            for path, error in corrupted:
                f.write(f"{path}\t{error}\n")
        print(f"Saved to {output_path}")


def main():
    args = parse_args()
    data_dir = args.data_dir
    if args.splits:
        for split in args.splits:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            output = data_dir / f"corrupted_{split}.txt"
            run_check(split_dir, split, output, args.black_threshold)
    else:
        output = data_dir / "corrupted_recursive.txt"
        run_check(data_dir, str(data_dir), output, args.black_threshold)

if __name__ == "__main__":
    main()
