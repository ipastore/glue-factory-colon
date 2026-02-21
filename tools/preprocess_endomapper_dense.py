"""Preprocess Endomapper sequences into per-sequence NPZ caches."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from gluefactory.datasets.endomapper_utils import (
    compute_overlap_matrix,
    extract_cameras_npz,
    extract_intrinsics,
    extract_poses,
    read_cameras_txt,
    read_images_txt,
    read_points3D_txt,
)

from gluefactory.settings import DATA_PATH

DEFAULT_ROOT = DATA_PATH / "endomapper_dense"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Endomapper COLMAP densified with Lightdepth outputs into NPZ caches."
    )

    parser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_ROOT,
            help="Root folder containing Seq_XXX_Y sequences.",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to place per-sequence scene_info NPZ files (default: <root>/scene_info).",
    )
    parser.add_argument(
        "--image-subpath",
        type=str,
        default="Undistorted_SfM",
        help="Root-relative image subpath saved into scene_info paths.",
    )
    parser.add_argument(
        "--depth-subpath",
        type=str,
        default="depth_undistorted",
        help="Root-relative depth subpath saved into scene_info paths.",
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


def _find_sequences(root: Path, image_subpath: str, names: List[str] | None) -> List[Path]:
    seq_root = root / image_subpath
    if names:
        return [seq_root / n for n in names]
    return sorted(p for p in seq_root.iterdir() if p.is_dir() and p.name.startswith("Seq_"))


def _find_map_ids(seq_dir: Path, map_ids: List[str] | None) -> List[str]:
    maps_root = seq_dir
    if map_ids is not None:
        return [str(mid) for mid in map_ids]
    if not maps_root.exists():
        return []
    return sorted([p.name for p in maps_root.iterdir() if p.is_dir()])


def _load_sequence_colmap(seq_dir: Path, map_id: str):
    colmap_dir = seq_dir / str(map_id) / "results_txt"
    cameras = read_cameras_txt(colmap_dir / "cameras.txt")
    images = read_images_txt(colmap_dir / "images.txt")
    points3d = read_points3D_txt(colmap_dir / "points3D.txt")
    return cameras, images, points3d

def _collect_point3d_arrays(points3d: Dict[int, object]) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(points3d.keys()), dtype=np.int64)
    coords = np.stack([points3d[i].xyz for i in ids], axis=0) if len(ids) > 0 else np.zeros((0, 3))
    return ids, coords


def process_sequence(
    seq_dir: Path,
    map_id: str,
    image_subpath: str,
    depth_subpath: str,
    out_dir: Path,
) -> Path:
    tqdm.write(f"[map] {seq_dir.name} map {map_id}: reading COLMAP")
    cameras, images, points3d = _load_sequence_colmap(seq_dir, map_id)

    image_ids = sorted(images.keys())
    image_names: List[str] = []
    image_camera_ids: List[int] = []
    point3d_ids_list: List[np.ndarray] = []
    for image_id in tqdm(
        image_ids,
        desc=f"{seq_dir.name} map {map_id} images",
        unit="img",
        leave=False,
    ):
        image = images[image_id]
        image_names.append(image.name)
        image_camera_ids.append(image.camera_id)
        point3d_ids_list.append(image.point3D_ids)

    image_sizes = np.array(
        [(cameras[cid].width, cameras[cid].height) for cid in image_camera_ids],
        dtype=np.int32,
    )
    intrinsics = extract_intrinsics(cameras, images)
    poses = extract_poses(images)
    cameras_npz, camera_indices = extract_cameras_npz(cameras, images)
    image_paths = np.array(
        [
            str(Path(image_subpath) / seq_dir.name / str(map_id) / "images" / Path(image_name).name)
            for image_name in image_names
        ],
        dtype=object,
    )
    depth_paths = np.array(
        [
            str(Path(depth_subpath) / seq_dir.name / str(map_id) / f"{Path(image_name).stem}_ttr.npz")
            for image_name in image_names
        ],
        dtype=object,
    )
    tqdm.write(f"[map] {seq_dir.name} map {map_id}: computing overlap matrix")
    overlap_matrix = compute_overlap_matrix(point3d_ids_list)
    point3d_ids_all, point3d_coords_all = _collect_point3d_arrays(points3d)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_dir.name}_map{map_id}.npz"
    tqdm.write(f"[map] {seq_dir.name} map {map_id}: saving {out_path.name}")
    np.savez(
        out_path,
        image_names=np.array(image_names, dtype=object),
        image_sizes=image_sizes,
        camera_ids=np.array(image_camera_ids, dtype=np.int64),
        cameras=cameras_npz,
        camera_indices=camera_indices,
        poses=poses,
        intrinsics=intrinsics,
        map_id=map_id,
        seq=seq_dir.name,
        overlap_matrix=overlap_matrix,
        point3D_ids_per_image=np.array(point3d_ids_list, dtype=object),
        point3D_ids=point3d_ids_all,
        point3D_coords=point3d_coords_all,
        image_paths=image_paths,
        depth_paths=depth_paths,
    )
    return out_path


def main():
    args = parse_args()
    root = args.root
    out_dir = args.output_dir if args.output_dir is not None else (root / "scene_info")


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
        sequences = [root / args.image_subpath / name for name in seq_order]
    else:
        sequences = _find_sequences(root, args.image_subpath, args.sequences)
    if not sequences:
        print(f"No sequences found under {root}")
        return 1

    # Unique pass: process sequences into NPZ
    print("=" * 60)
    print("Processing sequences into NPZ caches")
    print("=" * 60)
    
    map_tasks: List[Tuple[Path, str]] = []
    for seq_dir in sequences:
        map_ids = (
            seq_map_ids.get(seq_dir.name, [])
            if seq_map_ids is not None
            else _find_map_ids(seq_dir, None)
        )
        if not map_ids:
            print(f"[skip] {seq_dir.name}: no maps found")
            continue
        for map_id in map_ids:
            map_tasks.append((seq_dir, map_id))

    for seq_dir, map_id in tqdm(map_tasks, desc="Maps", unit="map"):
        out_path = out_dir / f"{seq_dir.name}_map{map_id}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {seq_dir.name} map {map_id} -> {out_path} (exists)")
            continue
        try:
            out = process_sequence(
                seq_dir,
                map_id,
                args.image_subpath,
                args.depth_subpath,
                out_dir,
            )
            print(f"[ok] {seq_dir.name} map {map_id} -> {out}")
        except Exception as exc:
            print(f"[fail] {seq_dir.name} map {map_id}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
