"""Preprocess Endomapper sequences into per-sequence NPZ caches."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gluefactory.datasets.endomapper_utils import (
    MISSING_DEPTH_VALUE,
    build_feature_depth_arrays,
    compute_overlap_matrix,
    extract_cameras_npz,
    extract_intrinsics,
    extract_poses,
    read_cameras_txt,
    read_depths_txt,
    read_features_txt,
    read_images_txt,
    read_points3D_txt,
)

from gluefactory.settings import DATA_PATH

DEFAULT_ROOT = DATA_PATH / "Endomapper_CUDASIFT"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Endomapper COLMAP outputs and CUDASIFT features into NPZ caches."
    )
    frames_group = parser.add_mutually_exclusive_group()
    frames_group.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Root directory containing video files (e.g., ~/all_sequences). If provided, extracts keyframes.",
    )
    frames_group.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Root directory containing per-frame extractions (Seq_*/Frame%08d.png). If provided, extracts keyframes.",
    )
    parser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_ROOT,
            help="Root folder containing Seq_XXX_Y sequences.",
        )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Directory to place per-sequence NPZ files (default: <root>/processed_npz).",
    )
    parser.add_argument(
        "--seq-maps",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of full map tokens (e.g., Seq_007_map148). Overrides --sequences.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of sequence folder names to process. Defaults to all Seq_* in root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ outputs.",
    )
    return parser.parse_args()

def _extract_frames_for_sequence(
    seq_dir: Path,
    frames_root: Path,
    out_root: Path,
    map_ids: List[str],
) -> bool:
    """Copy pre-extracted frames into each map keyframes directory."""
    seq_name = seq_dir.name
    frames_dir = frames_root / seq_name
    if not frames_dir.exists():
        print(f"  [warn] Frames not found: {frames_dir}")
        return False

    maps_root = seq_dir / "output" / "3D_maps"
    if not maps_root.exists():
        print(f"  [warn] No 3D_maps directory: {maps_root}")
        return False

    print(f"  [frames] Copying keyframes for {seq_name}...")
    numeric_name = re.compile(r"^[0-9]+$")
    copied = 0
    images_by_map: Dict[str, Dict[int, object]] = {}
    for map_id in map_ids:
        images_txt = maps_root / str(map_id) / "images.txt"
        if images_txt.exists():
            images_by_map[map_id] = read_images_txt(images_txt)
        else:
            images_by_map[map_id] = {}

    for map_id in map_ids:
        map_dir = maps_root / str(map_id)
        images = images_by_map.get(str(map_id))
        if not images:
            print(f"  [warn] No images loaded for map {map_id}")
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        for image in images.values():
            stem = Path(image.name).stem
            if not numeric_name.match(stem):
                print(f"  [warn] Non-numeric image name: {image.name}")
                continue
            n = int(stem)
            src = frames_dir / f"Frame{n:08d}.png"
            if not src.exists():
                alt = frames_dir / f"Frame{n:08d}.png"
                if alt.exists():
                    src = alt
                else:
                    print(f"  [warn] Missing frame: {src.name}")
                    continue
            shutil.copy2(src, out_root / f"{seq_name}_{n}.png")
            copied += 1

    if copied:
        print(f"  [frames] ✓ Copied {copied} frames for {seq_name}")
        return True
    print(f"  [frames] ✗ No frames copied for {seq_name}")
    return False

def _extract_frames_from_video(
    seq_dir: Path,
    video_root: Path,
    map_ids: List[str],
    out_root: Path,
) -> bool:
    """Extract frames for a sequence using extract_frames_depths_matches_endomapper_seq.py"""
    seq_name = seq_dir.name
    video_path = video_root / f"{seq_name}.mp4"

    if not video_path.exists():
        video_path = video_root / seq_name / f"{seq_name}.mov"

    if not video_path.exists():
        print(f"  [warn] Video not found: {video_path}")
        return False

    maps_root = seq_dir / "output" / "3D_maps"
    if not maps_root.exists():
        print(f"  [warn] No 3D_maps directory: {maps_root}")
        return False

    print(f"  [frames] Extracting keyframes for {seq_name}...")

    cmd = [
        "python3",
        "tools/extract_frames_depths_matches_endomapper_seq.py",
        "--maps_root", str(maps_root),
        "--keyframes_in_map_dir",
        "--video", str(video_path),
        "--number_from", "name",
        "--out_root", str(out_root),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"  [frames] ✓ Extracted frames for {seq_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [frames] ✗ Failed to extract frames: {e.stderr}")
        return False

def _find_sequences(root: Path, names: List[str] | None) -> List[Path]:
    if names:
        return [root / n for n in names]
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Seq_"))


def _find_map_ids(seq_dir: Path, map_ids: List[str] | None) -> List[str]:
    maps_root = seq_dir / "output" / "3D_maps"
    if map_ids is not None:
        return [str(mid) for mid in map_ids]
    if not maps_root.exists():
        return []
    return sorted([p.name for p in maps_root.iterdir() if p.is_dir()])



def main():
    args = parse_args()
    root = args.root
    out_root = args.out_root
    video_root = args.video_root
    frames_root = args.frames_root

    seq_map_ids = None
    if args.seq_maps:
        seq_map_ids = {}
        seq_order: List[str] = []
        for token in args.seq_maps:
            match = re.match(r"^(Seq_.*)_map(\d+)$", token)
            if not match:
                raise ValueError(
                    f"Invalid --seq-maps token '{token}'. Expected format: Seq_XXX_mapNNN"
                )
            seq_name, map_id = match.group(1), match.group(2)
            if seq_name not in seq_map_ids:
                seq_map_ids[seq_name] = []
                seq_order.append(seq_name)
            if map_id not in seq_map_ids[seq_name]:
                seq_map_ids[seq_name].append(map_id)
        sequences = [root / name for name in seq_order]
    else:
        sequences = _find_sequences(root, args.sequences)
    if not sequences:
        print(f"No sequences found under {root}")
        return 1

    print("=" * 60)
    if frames_root:
        print("Extracting keyframes from frames")
    else:
        print("Extracting keyframes from videos")
    print("=" * 60)
    for seq_dir in sequences:
        map_ids = (
            seq_map_ids.get(seq_dir.name, [])
            if seq_map_ids is not None
            else _find_map_ids(seq_dir, None)
        )
        if not map_ids:
            print(f"[skip] {seq_dir.name}: no maps under output/3D_maps/")
            continue
        if frames_root:
            _extract_frames_for_sequence(seq_dir, frames_root, out_root, map_ids)
        else:
            _extract_frames_from_video(seq_dir, video_root,  out_root, map_ids, root)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
