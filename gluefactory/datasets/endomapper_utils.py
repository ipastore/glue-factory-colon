"""Utilities to parse COLMAP exports for the Endomapper dataset."""

from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

FEATURE_DIM = 128
MISSING_DEPTH_VALUE = -1.0


class ColmapCamera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


class ColmapImage(NamedTuple):
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


class ColmapPoint3D(NamedTuple):
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[Tuple[int, int]]


def _read_valid_lines(path: Path) -> List[str]:
    """Return non-empty, non-comment lines."""
    path = Path(path)
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert from quaternions to rotation matrix (qw, qx, qy, qz)."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qw * qz,
                2 * qz * qx + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qz * qx - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ],
        dtype=np.float64,
    )


def read_cameras_txt(path: Path) -> Dict[int, ColmapCamera]:
    """Parse COLMAP cameras.txt into a dictionary keyed by camera id."""
    cameras: Dict[int, ColmapCamera] = {}
    for line in _read_valid_lines(path):
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError("Invalid cameras.txt line, expected id model width height.")
        camera_id = int(tokens[0])
        model = tokens[1]
        width, height = int(tokens[2]), int(tokens[3])
        params = np.array(list(map(float, tokens[4:])), dtype=np.float32)
        if model == "OPENCV_FISHEYE" and params.shape[0] != 8:
            raise ValueError("OPENCV_FISHEYE expects 8 params (fx, fy, cx, cy, k1-4).")
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    return cameras


def read_images_txt(path: Path) -> Dict[int, ColmapImage]:
    """Parse COLMAP images.txt into a dictionary keyed by image id."""
    lines = _read_valid_lines(path)
    if len(lines) % 2 != 0:
        raise ValueError("images.txt must contain pose and points2D lines in pairs.")

    images: Dict[int, ColmapImage] = {}
    for pose_line, points_line in zip(lines[0::2], lines[1::2]):
        pose_tokens = pose_line.split()
        if len(pose_tokens) < 9:
            raise ValueError("Invalid pose line in images.txt.")
        image_id = int(pose_tokens[0])
        qvec = np.array(list(map(float, pose_tokens[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, pose_tokens[5:8])), dtype=np.float64)
        camera_id = int(pose_tokens[8])
        name = pose_tokens[-1]

        points_tokens = points_line.split()
        if len(points_tokens) % 3 != 0:
            raise ValueError(
                f"Image {image_id} has incomplete (x, y, point3D_id) triplets."
            )
        xys, point3d_ids = [], []
        for i in range(0, len(points_tokens), 3):
            xys.append((float(points_tokens[i]), float(points_tokens[i + 1])))
            point3d_ids.append(int(points_tokens[i + 2]))
        xys_arr = np.array(xys, dtype=np.float64)
        point3d_ids_arr = np.array(point3d_ids, dtype=np.int64)

        images[image_id] = ColmapImage(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys_arr,
            point3D_ids=point3d_ids_arr,
        )
    return images


def read_points3D_txt(path: Path) -> Dict[int, ColmapPoint3D]:
    """Parse COLMAP points3D.txt into a dictionary keyed by point3D id."""
    points3d: Dict[int, ColmapPoint3D] = {}
    for line in _read_valid_lines(path):
        tokens = line.split()
        if len(tokens) < 9:
            raise ValueError("Invalid points3D.txt line, expected track entries.")
        point3d_id = int(tokens[0])
        xyz = np.array(list(map(float, tokens[1:4])), dtype=np.float64)
        rgb = np.array(list(map(int, tokens[4:7])), dtype=np.int64)
        error = float(tokens[7])
        track_tokens = tokens[8:]
        if len(track_tokens) % 2 != 0:
            raise ValueError(f"Point3D {point3d_id} track has dangling entries.")
        track: List[Tuple[int, int]] = []
        for i in range(0, len(track_tokens), 2):
            track.append((int(track_tokens[i]), int(track_tokens[i + 1])))
        points3d[point3d_id] = ColmapPoint3D(
            id=point3d_id, xyz=xyz, rgb=rgb, error=error, track=track
        )
    return points3d


def _camera_fx_fy_cx_cy(camera: ColmapCamera) -> Tuple[float, float, float, float]:
    if camera.params.shape[0] < 4:
        raise ValueError(f"Camera {camera.id} params missing fx, fy, cx, cy.")
    fx, fy, cx, cy = camera.params[:4]
    return np.float32(fx), np.float32(fy), np.float32(cx), np.float32(cy)


def extract_intrinsics(
    cameras: Dict[int, ColmapCamera], images: Dict[int, ColmapImage]
) -> np.ndarray:
    """Return [N,3,3] intrinsics ordered by sorted image id."""
    Ks = []
    for image_id in sorted(images.keys()):
        image = images[image_id]
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = _camera_fx_fy_cx_cy(camera)
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1] = fx, fy
        K[0, 2], K[1, 2] = cx, cy
        Ks.append(K)
    return np.stack(Ks, axis=0)


def extract_poses(images: Dict[int, ColmapImage]) -> np.ndarray:
    """Return [N,4,4] poses T_w2c ordered by sorted image id."""
    poses = []
    for image_id in sorted(images.keys()):
        image = images[image_id]
        # use float32 to match downstream PyTorch tensors (GPU-friendly)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = qvec2rotmat(image.qvec).astype(np.float32)
        T[:3, 3] = image.tvec.astype(np.float32)
        poses.append(T)
    return np.stack(poses, axis=0)


def read_features_txt(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """Parse features file with rows: KPID, X, Y, SCALE, ORI, SCORE, DESC[128]."""
    features: Dict[int, Dict[str, np.ndarray]] = {}
    for line in _read_valid_lines(path):
        tokens = line.split()
        if len(tokens) < 6 + FEATURE_DIM:
            raise ValueError(
                f"Feature line for KPID {kpid} missing values "
                f"(expected {6 + FEATURE_DIM}, got {len(tokens)})."
            )
        
        kpid = int(tokens[0])
        xy = np.array([float(tokens[1]), float(tokens[2])], dtype=np.float32)
        scale = float(tokens[3])
        orientation = float(tokens[4])
        score = float(tokens[5])
        desc_vals = list(map(float, tokens[6 : 6 + FEATURE_DIM]))
        if len(desc_vals) != FEATURE_DIM:
            raise ValueError(
                f"Descriptor length for KPID {kpid} is {len(desc_vals)}, "
                f"expected {FEATURE_DIM}."
            )
        descriptor = np.array(desc_vals, dtype=np.float32)
        features[kpid] = {
            "xy": xy,
            "scale": np.float32(scale),
            "orientation": np.float32(orientation),
            "score": np.float32(score),
            "descriptor": descriptor,
        }
    return features


def read_depths_txt(path: Path) -> Dict[int, float]:
    """Parse depths file with rows: KPID, DEPTH."""
    depths: Dict[int, float] = {}
    for line in _read_valid_lines(path):
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(f"Depth line for line {line} missing depth value.")
        
        kpid = int(tokens[0])
        depth_val = np.float32(tokens[1])
        depths[kpid] = depth_val
    return depths


def build_feature_depth_arrays(
    features: Dict[int, Dict[str, np.ndarray]], depths: Dict[int, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted KPID list and aligned arrays (keypoints, descriptors, depths, scale, orientation, score)."""
    if not features or not depths:
        raise ValueError("Cannot build _depth arrays: features or depths dictionary is empty.")

    kpids = np.array(sorted(features.keys()), dtype=np.int64)
    keypoints = np.zeros((len(kpids), 2), dtype=np.float32)
    descriptors = np.zeros((len(kpids), FEATURE_DIM), dtype=np.float32)
    depth_values = np.full((len(kpids),), MISSING_DEPTH_VALUE, dtype=np.float32)
    scales = np.zeros((len(kpids),), dtype=np.float32)
    orientations = np.zeros((len(kpids),), dtype=np.float32)
    scores = np.zeros((len(kpids),), dtype=np.float32)

    for idx, kpid in enumerate(kpids):
        feat = features[kpid]
        keypoints[idx] = feat["xy"]
        desc = feat["descriptor"]
        if desc.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"Descriptor for KPID {kpid} has shape {desc.shape}, "
                f"expected ({FEATURE_DIM},)."
            )
        descriptors[idx] = desc
        depth_values[idx] = np.float32(depths.get(kpid, MISSING_DEPTH_VALUE))
        scales[idx] = feat["scale"]
        orientations[idx] = feat["orientation"]
        scores[idx] = feat["score"]

    return kpids, keypoints, descriptors, depth_values, scales, orientations, scores


def compute_overlap_matrix(
    point3d_ids_list: List[np.ndarray],
) -> np.ndarray:
    """
    Compute overlap matrix using |A∩B|/min(|A|,|B|) over valid point3D ids.

    point3d_ids_list: list of per-image arrays; -1 denotes invalid.
    """
    n = len(point3d_ids_list)
    overlap = np.zeros((n, n), dtype=np.float32)

    valid_sets = [
        set(int(pid) for pid in ids if pid != -1)
        for ids in point3d_ids_list
    ]

    for i in range(n):
        for j in range(i, n):
            denom = min(len(valid_sets[i]), len(valid_sets[j]))
            if denom == 0:
                value = 0.0
            else:
                value = len(valid_sets[i] & valid_sets[j]) / denom
            overlap[i, j] = overlap[j, i] = value
    return overlap


__all__ = [
    "ColmapCamera",
    "ColmapImage",
    "ColmapPoint3D",
    "FEATURE_DIM",
    "MISSING_DEPTH_VALUE",
    "qvec2rotmat",
    "read_cameras_txt",
    "read_images_txt",
    "read_points3D_txt",
    "extract_intrinsics",
    "extract_poses",
    "read_features_txt",
    "read_depths_txt",
    "build_feature_depth_arrays",
    "compute_overlap_matrix",
]
