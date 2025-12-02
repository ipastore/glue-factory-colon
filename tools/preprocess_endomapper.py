"""Preprocess Endomapper sequences into per-sequence NPZ caches."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gluefactory.datasets.endomapper_utils import (
    MISSING_DEPTH_VALUE,
    build_feature_depth_arrays,
    compute_overlap_matrix,
    extract_intrinsics,
    extract_poses,
    read_cameras_txt,
    read_depths_txt,
    read_features_txt,
    read_images_txt,
    read_points3D_txt,
)

# Default root from AGENTS instructions
# DEFAULT_ROOT = Path(
#     "/media/student/HDD/nacho/glue-factory/data/Endomapper_CUDASIFT_OCT25"
# )

DEFAULT_ROOT = Path(
    "/home/student/glue-factory-colon/data/Endomapper_CUDASIFT_NOV25"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Endomapper COLMAP outputs and CUDASIFT features into NPZ caches."
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
        help="Directory to place per-sequence NPZ files (default: <root>/processed_npz).",
    )
    parser.add_argument(
        "--map-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of map ids to process (default: all under output/3D_maps/).",
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


def _load_sequence_colmap(seq_dir: Path, map_id: str):
    colmap_dir = seq_dir / "output" / "3D_maps" / str(map_id)
    cameras = read_cameras_txt(colmap_dir / "cameras.txt")
    images = read_images_txt(colmap_dir / "images.txt")
    points3d = read_points3D_txt(colmap_dir / "points3D.txt")
    return cameras, images, points3d


def _distortion_coeffs(camera_model: str, params: np.ndarray) -> np.ndarray:
    if camera_model == "OPENCV_FISHEYE" and params.shape[0] >= 8:
        return params[4:8].astype(np.float64)
    return np.zeros((4,), dtype=np.float64)


def _collect_point3d_arrays(points3d: Dict[int, object]) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(points3d.keys()), dtype=np.int64)
    coords = np.stack([points3d[i].xyz for i in ids], axis=0) if len(ids) > 0 else np.zeros((0, 3))
    return ids, coords


def process_sequence(
    seq_dir: Path,
    map_id: str,
    out_dir: Path,
) -> Path:
    cameras, images, points3d = _load_sequence_colmap(seq_dir, map_id)

    image_ids = sorted(images.keys())
    image_names = [images[i].name for i in image_ids]
    camera_model = cameras[images[image_ids[0]].camera_id].model

    intrinsics = extract_intrinsics(cameras, images)
    poses = extract_poses(images)
    distortion_coeffs = []
    for image_id in image_ids:
        image = images[image_id]
        camera = cameras[image.camera_id]
        distortion_coeffs.append(_distortion_coeffs(camera.model, camera.params))

    features_dir = seq_dir / f"output/3D_maps/{map_id}/features"
    depths_dir = seq_dir / f"output/3D_maps/{map_id}/depths"

    keypoints_list: List[np.ndarray] = []
    descriptors_list: List[np.ndarray] = []
    depths_list: List[np.ndarray] = []
    scales_list: List[np.ndarray] = []
    orientations_list: List[np.ndarray] = []
    scores_list: List[np.ndarray] = []
    point3d_ids_list: List[np.ndarray] = []
    valid_depth_mask_list: List[np.ndarray] = []
    valid_3d_mask_list: List[np.ndarray] = []

    for image_id in image_ids:
        image = images[image_id]
        stem = Path(image.name).stem
        feat_path = features_dir / f"{stem}_features.txt"
        depth_path = depths_dir / f"{stem}_depths.txt"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature file missing for image {image.name}: {feat_path}"
            )
        if not depth_path.exists():
            raise FileNotFoundError(
                f"Depth file missing for image {image.name}: {depth_path}"
            )     
        feature_data = read_features_txt(feat_path)
        depth_data = read_depths_txt(depth_path)
        
        (
            kpids,
            keypoints,
            descriptors,
            depth_values,
            scales,
            orientations,
            scores,
        ) = build_feature_depth_arrays(feature_data, depth_data)

        point3d_ids = image.point3D_ids

        valid_depth_mask = depth_values != MISSING_DEPTH_VALUE
        valid_3d_mask = point3d_ids != -1

        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        depths_list.append(depth_values)
        scales_list.append(scales)
        orientations_list.append(orientations)
        scores_list.append(scores)
        point3d_ids_list.append(point3d_ids)
        valid_depth_mask_list.append(valid_depth_mask)
        valid_3d_mask_list.append(valid_3d_mask)
    
    

    overlap_matrix = compute_overlap_matrix(point3d_ids_list)
    point3d_ids_all, point3d_coords_all = _collect_point3d_arrays(points3d)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_dir.name}_map{map_id}.npz"
    np.savez(
        out_path,
        image_names=np.array(image_names, dtype=object),
        poses=poses,
        intrinsics=intrinsics,
        distortion_coeffs=np.stack(distortion_coeffs, axis=0),
        camera_model=camera_model,
        map_id=map_id,
        seq=seq_dir.name,
        overlap_matrix=overlap_matrix,
        keypoints_per_image=np.array(keypoints_list, dtype=object),
        descriptors_per_image=np.array(descriptors_list, dtype=object),
        depths_per_image=np.array(depths_list, dtype=object),
        scales_per_image=np.array(scales_list, dtype=object),
        orientations_per_image=np.array(orientations_list, dtype=object),
        scores_per_image=np.array(scores_list, dtype=object),
        point3D_ids_per_image=np.array(point3d_ids_list, dtype=object),
        valid_depth_mask_per_image=np.array(valid_depth_mask_list, dtype=object),
        valid_3d_mask_per_image=np.array(valid_3d_mask_list, dtype=object),
        point3D_ids=point3d_ids_all,
        point3D_coords=point3d_coords_all,
    )
    return out_path


def main():
    args = parse_args()
    root = args.root
    out_dir = args.output_dir or (root / "processed_npz")

    sequences = _find_sequences(root, args.sequences)
    if not sequences:
        print(f"No sequences found under {root}")
        return 1

    for seq_dir in sequences:
        map_ids = _find_map_ids(seq_dir, args.map_ids)
        if not map_ids:
            print(f"[skip] {seq_dir.name}: no maps under output/3D_maps/")
            continue
        for map_id in map_ids:
            out_path = out_dir / f"{seq_dir.name}_map{map_id}.npz"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {seq_dir.name} map {map_id} -> {out_path} (exists)")
                continue
            try:
                out = process_sequence(seq_dir, map_id, out_dir)
                print(f"[ok] {seq_dir.name} map {map_id} -> {out}")
            except Exception as exc:
                print(f"[fail] {seq_dir.name} map {map_id}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
