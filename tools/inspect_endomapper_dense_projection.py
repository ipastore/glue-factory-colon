#!/usr/bin/env python3
"""Inspect EndomapperDense two-view sparse reprojection and GT correspondences."""

from __future__ import annotations

import argparse
import colorsys
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from gluefactory.datasets.endomapper_dense import _PairDataset
from gluefactory.datasets.endomapper_utils import read_images_txt
from gluefactory.geometry.depth import project, sample_depth
from gluefactory.geometry.wrappers import Pose
from gluefactory.models.extractors.sift import SIFT
from gluefactory.settings import DATA_PATH
from gluefactory.utils.image import ImagePreprocessor, load_image


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect EndomapperDense projections for a user-selected pair. "
            "Uses COLMAP correspondences and CUDA-SIFT features."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_PATH / "endomapper_dense",
        help="Root folder of endomapper_dense dataset.",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene token, e.g. Seq_008_a_map0.",
    )
    parser.add_argument(
        "--image-a",
        required=True,
        help="First image name token (exact name, basename, or stem).",
    )
    parser.add_argument(
        "--image-b",
        required=True,
        help="Second image name token (exact name, basename, or stem).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/endomapper_dense_projection"),
        help="Output directory for overlays and metrics.",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=2000,
        help="Maximum correspondences to draw per overlay (0 means all).",
    )
    parser.add_argument(
        "--coverage-radius",
        type=float,
        default=3.0,
        help="Pixel radius used for GT-vs-SIFT coverage statistics.",
    )
    parser.add_argument(
        "--random-sift-count",
        type=int,
        default=10,
        help="Number of random CUDA-SIFT keypoints from view A to reproject into view B.",
    )
    parser.add_argument(
        "--random-sift-seed",
        type=int,
        default=0,
        help="Random seed used to sample CUDA-SIFT keypoints for reprojection debugging.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_scene(scene: str) -> tuple[str, str]:
    match = re.match(r"^(Seq_.*)_map(\d+)$", scene)
    if not match:
        raise ValueError(
            f"Invalid scene '{scene}'. Expected format: Seq_XXX_Y_mapN."
        )
    return match.group(1), match.group(2)


def _resolve_image_index(image_names: np.ndarray, token: str) -> int:
    names = [str(x) for x in image_names.tolist()]
    token_base = Path(token).name
    token_stem = Path(token_base).stem
    for i, name in enumerate(names):
        name_base = Path(name).name
        if token in {name, name_base, Path(name_base).stem}:
            return i
        if token_base in {name, name_base, Path(name_base).stem}:
            return i
        if token_stem in {name, name_base, Path(name_base).stem}:
            return i
    raise ValueError(
        f"Cannot resolve image token '{token}'. Available sample names: {names[:5]}"
    )


def _make_loader(data_root: Path, scene: str):
    try:
        rel = data_root.resolve().relative_to(DATA_PATH.resolve())
    except ValueError as exc:
        raise ValueError(
            f"--data-root must be inside DATA_PATH ({DATA_PATH}), got {data_root}."
        ) from exc

    conf = OmegaConf.create(
        {
            "data_dir": str(rel).rstrip("/") + "/",
            "depth_subpath": "depth_undistorted/",
            "image_subpath": "Undistorted_SfM/",
            "info_dir": "scene_info/",
            "views": 2,
            "read_depth": True,
            "read_image": True,
            "grayscale": False,
            "p_rotate": 0.0,
            "reseed": False,
            "seed": 0,
            "preprocessing": ImagePreprocessor.default_conf,
            "load_features": {"do": False},
            "val_split": [scene],
            "val_num_per_scene": None,
            "val_pairs": None,
            "train_split": [scene],
            "train_num_per_scene": None,
            "test_split": [scene],
            "test_num_per_scene": None,
            "test_pairs": None,
            "min_overlap": 0.0,
            "max_overlap": 1.0,
            "num_overlap_bins": 1,
            "sort_by_overlap": False,
            "triplet_enforce_overlap": False,
        }
    )
    return _PairDataset(conf, "val", load_sample=False)


def _obs_by_id(colmap_image) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for xy, pid in zip(colmap_image.xys, colmap_image.point3D_ids):
        if pid == -1:
            continue
        pid_i = int(pid)
        if pid_i not in out:
            out[pid_i] = np.asarray(xy, dtype=np.float32)
    return out


def _crop_points(uv: np.ndarray, left_top: Tuple[float, float], w: int, h: int):
    uv_c = uv.copy()
    uv_c[:, 0] -= left_top[0]
    uv_c[:, 1] -= left_top[1]
    valid = (
        (uv_c[:, 0] >= 0.0)
        & (uv_c[:, 0] <= (w - 1))
        & (uv_c[:, 1] >= 0.0)
        & (uv_c[:, 1] <= (h - 1))
    )
    return uv_c, valid


def _sample_indices(n: int, max_draw: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    if max_draw > 0 and n > max_draw:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_draw, replace=False))
    return idx


def _to_bgr_image(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _to_gray_luminance(img_t: torch.Tensor) -> torch.Tensor:
    # Luma coefficients for RGB input in [0, 1].
    r = img_t[0:1]
    g = img_t[1:2]
    b = img_t[2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _depth_to_colorbar_image(depth_t: torch.Tensor) -> np.ndarray:
    depth = depth_t.detach().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0.0)
    if np.any(valid):
        dmin = float(np.min(depth[valid]))
        dmax = float(np.max(depth[valid]))
    else:
        dmin, dmax = 0.0, 1.0

    denom = max(dmax - dmin, 1e-6)
    depth_n = np.clip((depth - dmin) / denom, 0.0, 1.0)
    depth_u8 = np.round(depth_n * 255.0).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_bgr[~valid] = (0, 0, 0)

    return depth_bgr


def _depth_overlay_image(img_bgr: np.ndarray, depth_t: torch.Tensor) -> np.ndarray:
    depth = depth_t.detach().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0.0)
    if np.any(valid):
        dmin = float(np.min(depth[valid]))
        dmax = float(np.max(depth[valid]))
    else:
        dmin, dmax = 0.0, 1.0

    denom = max(dmax - dmin, 1e-6)
    depth_n = np.clip((depth - dmin) / denom, 0.0, 1.0)
    depth_u8 = np.round(depth_n * 255.0).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    out = img_bgr.copy()
    overlay = cv2.addWeighted(img_bgr, 0.5, depth_bgr, 0.5, 0.0)
    out[valid] = overlay[valid]
    return out


def _hstack_images(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] :] = right
    return out


def _draw_rainbow_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    max_draw: int,
) -> np.ndarray:
    canvas = _hstack_images(img_a, img_b)
    shift = img_a.shape[1]
    idx = _sample_indices(len(pts_a), max_draw)
    n = max(1, len(idx))
    for rank, i in enumerate(idx):
        hue = rank / n
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bgr = tuple(int(round(c * 255.0)) for c in rgb[::-1])
        ax, ay = pts_a[i]
        bx, by = pts_b[i]
        p0 = (int(round(ax)), int(round(ay)))
        p1 = (int(round(bx + shift)), int(round(by)))
        cv2.circle(canvas, p0, 4, bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, bgr, -1, cv2.LINE_AA)
    return canvas


def _draw_residual_overlay(
    img: np.ndarray,
    measured: np.ndarray,
    projected: np.ndarray,
    errors: np.ndarray,
    max_draw: int,
) -> np.ndarray:
    out = img.copy()
    idx = _sample_indices(len(measured), max_draw)
    emax = float(np.max(errors[idx])) if len(idx) > 0 else 1.0
    emax = max(emax, 1e-6)
    for i in idx:
        um, vm = measured[i]
        up, vp = projected[i]
        t = float(np.clip(errors[i] / emax, 0.0, 1.0))
        c = colorsys.hsv_to_rgb((1.0 - t) * 0.33, 1.0, 1.0)
        bgr = tuple(int(round(v * 255.0)) for v in c[::-1])
        p_m = (int(round(um)), int(round(vm)))
        p_p = (int(round(up)), int(round(vp)))
        cv2.line(out, p_p, p_m, bgr, 1, cv2.LINE_AA)
        cv2.circle(out, p_m, 2, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p_p, 2, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def _extract_cudasift_keypoints(gray: torch.Tensor) -> np.ndarray:
    try:
        sift = SIFT(
            {
                "name": "extractors.sift",
                "backend": "py_cudasift",
                "max_num_keypoints": 4096,
                "nms_radius": 0,
                "rootsift": True,
                "force_num_keypoints": False,
            }
        )
    except ImportError as exc:
        raise ImportError(
            "py_cudasift backend is required for this tool. "
            "Install/build cudasift_py in this environment."
        ) from exc
    pred = sift.extract_single_image(gray.float().cpu())
    return pred["keypoints"].cpu().numpy().astype(np.float32, copy=False)


def _nearest_dist(queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    if len(queries) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(refs) == 0:
        return np.full((len(queries),), np.inf, dtype=np.float32)
    best = np.full((len(queries),), np.inf, dtype=np.float32)
    block = 4096
    for i in range(0, len(refs), block):
        r = refs[i : i + block]
        d = queries[:, None, :] - r[None, :, :]
        d2 = np.sum(d * d, axis=-1)
        best = np.minimum(best, np.sqrt(np.min(d2, axis=1)))
    return best


def _draw_coverage_overlay(
    img: np.ndarray, gt_pts: np.ndarray, sift_pts: np.ndarray, radius: float
) -> np.ndarray:
    out = img.copy()
    d = _nearest_dist(gt_pts, sift_pts)
    covered = d <= radius
    for p in sift_pts:
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 1, (0, 255, 0), -1)
    for i, p in enumerate(gt_pts):
        color = (0, 255, 255) if covered[i] else (0, 0, 255)
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 2, color, -1)
    return out


def _stats(errors: np.ndarray) -> dict:
    if len(errors) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(errors)),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def _inlier_rates(errors: np.ndarray, thresholds: np.ndarray) -> dict:
    if len(errors) == 0:
        return {float(t): float("nan") for t in thresholds}
    return {float(t): float(np.mean(errors <= t)) for t in thresholds}


def _summary_stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _lookup_xyz(scene_info: Dict, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids_all = scene_info["point3D_ids"].astype(np.int64)
    xyz_all = scene_info["point3D_coords"].astype(np.float32)
    if len(ids_all) == 0:
        return np.zeros((len(point_ids), 3), dtype=np.float32), np.zeros(
            (len(point_ids),), dtype=bool
        )
    order = np.argsort(ids_all)
    ids_sorted = ids_all[order]
    xyz_sorted = xyz_all[order]
    q = point_ids.astype(np.int64)
    pos = np.searchsorted(ids_sorted, q)
    ok = (pos >= 0) & (pos < ids_sorted.shape[0]) & (ids_sorted[pos] == q)
    xyz = np.zeros((q.shape[0], 3), dtype=np.float32)
    if np.any(ok):
        xyz[ok] = xyz_sorted[pos[ok]]
    return xyz, ok


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root
    scene_info_path = data_root / "scene_info" / f"{args.scene}.npz"
    if not scene_info_path.exists():
        raise FileNotFoundError(f"scene_info file not found: {scene_info_path}")

    seq, map_id = _parse_scene(args.scene)
    colmap_dir = (
        data_root
        / "Undistorted_SfM"
        / seq
        / map_id
        / "results_txt"
    )
    images_txt = colmap_dir / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP txt files in {colmap_dir}")

    with np.load(str(scene_info_path), allow_pickle=True) as scene_info_npz:
        scene_info = {k: scene_info_npz[k] for k in scene_info_npz.files}

    pair_loader = _make_loader(data_root, args.scene)
    scene = args.scene

    image_names = scene_info["image_names"]
    idx_a = _resolve_image_index(image_names, args.image_a)
    idx_b = _resolve_image_index(image_names, args.image_b)
    name_a = str(image_names[idx_a])
    name_b = str(image_names[idx_b])

    images = read_images_txt(images_txt)
    by_name = {v.name: v for v in images.values()}
    if name_a not in by_name or name_b not in by_name:
        raise KeyError(
            f"Pair names not present in images.txt: '{name_a}', '{name_b}'."
        )
    im_a = by_name[name_a]
    im_b = by_name[name_b]

    obs_a = _obs_by_id(im_a)
    obs_b = _obs_by_id(im_b)
    shared_ids_raw = sorted(set(obs_a.keys()) & set(obs_b.keys()))
    if len(shared_ids_raw) == 0:
        raise ValueError(f"No shared point3D_ids between '{name_a}' and '{name_b}'.")

    pts_a_raw = np.stack([obs_a[pid] for pid in shared_ids_raw], axis=0)
    pts_b_raw = np.stack([obs_b[pid] for pid in shared_ids_raw], axis=0)

    if not pair_loader.valid[scene][idx_a] or not pair_loader.valid[scene][idx_b]:
        raise ValueError(
            "Selected pair is not valid for dataloader (missing image/depth). "
            f"idx_a={idx_a}, idx_b={idx_b}."
        )

    view_a = pair_loader._read_view(scene, idx_a)
    view_b = pair_loader._read_view(scene, idx_b)
    img_a = view_a["image"]
    img_b = view_b["image"]
    depth_a_t = view_a["depth"][None]
    depth_b_t = view_b["depth"][None]
    cam_a = view_a["camera"]
    cam_b = view_b["camera"]
    T_w2a = view_a["T_w2cam"]
    T_w2b = view_b["T_w2cam"]

    raw_img_a = load_image(data_root / str(scene_info["image_paths"][idx_a]), grayscale=False)
    raw_img_b = load_image(data_root / str(scene_info["image_paths"][idx_b]), grayscale=False)
    pre = ImagePreprocessor({})
    _, off_a = pre.crop_endomapper_dense(raw_img_a)
    _, off_b = pre.crop_endomapper_dense(raw_img_b)
    if off_a != off_b:
        raise ValueError(f"Unexpected different crop offsets: {off_a} vs {off_b}")
    h, w = img_a.shape[-2:]

    pts_a, in_a = _crop_points(pts_a_raw, off_a, w, h)
    pts_b, in_b = _crop_points(pts_b_raw, off_b, w, h)
    in_crop = in_a & in_b
    ids_crop = np.asarray(
        [pid for keep, pid in zip(in_crop, shared_ids_raw) if keep], dtype=np.int64
    )
    pts_a = pts_a[in_crop]
    pts_b = pts_b[in_crop]

    if tuple(depth_a_t.shape[-2:]) != (h, w) or tuple(depth_b_t.shape[-2:]) != (h, w):
        raise ValueError("Dataloader depth size does not match dataloader image size.")

    T_a2b = T_w2b @ T_w2a.inv()
    T_b2a = T_w2a @ T_w2b.inv()
    pts_a_t = torch.from_numpy(pts_a)[None].float()
    pts_b_t = torch.from_numpy(pts_b)[None].float()

    dr, dt = T_a2b.magnitude()
    pose_dt = float(dt.detach().cpu().item())
    pose_dr = float(dr.detach().cpu().item())

    d_a, valid_a = sample_depth(pts_a_t, depth_a_t)
    d_b, valid_b = sample_depth(pts_b_t, depth_b_t)

    valid_a_count = int(valid_a.sum().detach().cpu().item())
    valid_b_count = int(valid_b.sum().detach().cpu().item())
    shared_count = int(pts_a.shape[0])

    scale_a = 1.0
    scale_b = 1.0
    scale_a_med = float("nan")
    scale_b_med = float("nan")
    scale_mode = "colmap_median"
    z_stats_a = _summary_stats(np.asarray([], dtype=np.float32))
    z_stats_b = _summary_stats(np.asarray([], dtype=np.float32))
    ratio_stats_a = _summary_stats(np.asarray([], dtype=np.float32))
    ratio_stats_b = _summary_stats(np.asarray([], dtype=np.float32))
    ratios_a = np.zeros((0,), dtype=np.float32)
    ratios_b = np.zeros((0,), dtype=np.float32)
    if shared_count > 0:
        xyz_world, xyz_ok = _lookup_xyz(scene_info, ids_crop)
        if np.any(xyz_ok):
            Xw = torch.from_numpy(xyz_world[xyz_ok]).float()
            Za = (T_w2a * Xw)[:, 2].detach().cpu().numpy()
            Zb = (T_w2b * Xw)[:, 2].detach().cpu().numpy()
            da_np = d_a[0].detach().cpu().numpy()[xyz_ok]
            db_np = d_b[0].detach().cpu().numpy()[xyz_ok]
            va_np = valid_a[0].detach().cpu().numpy()[xyz_ok]
            vb_np = valid_b[0].detach().cpu().numpy()[xyz_ok]

            good_a = va_np & np.isfinite(da_np) & (da_np > 0) & np.isfinite(Za) & (Za > 1e-6)
            good_b = vb_np & np.isfinite(db_np) & (db_np > 0) & np.isfinite(Zb) & (Zb > 1e-6)
            if np.any(good_a):
                ratios_a = Za[good_a] / da_np[good_a]
                z_stats_a = _summary_stats(Za[good_a])
                ratio_stats_a = _summary_stats(ratios_a)
                scale_a_med = float(np.median(ratios_a))
                scale_a = scale_a_med
            if np.any(good_b):
                ratios_b = Zb[good_b] / db_np[good_b]
                z_stats_b = _summary_stats(Zb[good_b])
                ratio_stats_b = _summary_stats(ratios_b)
                scale_b_med = float(np.median(ratios_b))
                scale_b = scale_b_med

    d_a_used = d_a * float(scale_a)
    d_b_used = d_b * float(scale_b)
    d_a_np = d_a[0].detach().cpu().numpy()
    d_b_np = d_b[0].detach().cpu().numpy()
    d_a_used_np = d_a_used[0].detach().cpu().numpy()
    d_b_used_np = d_b_used[0].detach().cpu().numpy()
    va_all = valid_a[0].detach().cpu().numpy().astype(bool)
    vb_all = valid_b[0].detach().cpu().numpy().astype(bool)
    good_depth_a = va_all & np.isfinite(d_a_np) & (d_a_np > 0)
    good_depth_b = vb_all & np.isfinite(d_b_np) & (d_b_np > 0)
    depth_stats_a_raw = _summary_stats(d_a_np[good_depth_a])
    depth_stats_b_raw = _summary_stats(d_b_np[good_depth_b])
    depth_stats_a_aligned = _summary_stats(d_a_used_np[good_depth_a])
    depth_stats_b_aligned = _summary_stats(d_b_used_np[good_depth_b])

    uv_b_proj_t, vis_b_t = project(
        pts_a_t, d_a_used, depth_b_t, cam_a, cam_b, T_a2b, valid_a, ccth=None
    )
    uv_a_proj_t, vis_a_t = project(
        pts_b_t, d_b_used, depth_a_t, cam_b, cam_a, T_b2a, valid_b, ccth=None
    )

    uv_a_proj = uv_a_proj_t[0].detach().cpu().numpy()
    uv_b_proj = uv_b_proj_t[0].detach().cpu().numpy()
    vis_a = vis_a_t[0].detach().cpu().numpy().astype(bool)
    vis_b = vis_b_t[0].detach().cpu().numpy().astype(bool)

    err_ab = np.linalg.norm(uv_b_proj - pts_b, axis=1)
    err_ba = np.linalg.norm(uv_a_proj - pts_a, axis=1)
    err_ab_v = err_ab[vis_b]
    err_ba_v = err_ba[vis_a]

    thresholds = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    stats_ab = _stats(err_ab_v)
    stats_ba = _stats(err_ba_v)
    inliers_ab = _inlier_rates(err_ab_v, thresholds)
    inliers_ba = _inlier_rates(err_ba_v, thresholds)

    gray_a = _to_gray_luminance(img_a)
    gray_b = _to_gray_luminance(img_b)
    kpts_a = _extract_cudasift_keypoints(gray_a)
    kpts_b = _extract_cudasift_keypoints(gray_b)
    dist_gt_a = _nearest_dist(pts_a, kpts_a)
    dist_gt_b = _nearest_dist(pts_b, kpts_b)
    cov_a = float(np.mean(dist_gt_a <= args.coverage_radius)) if len(dist_gt_a) else float("nan")
    cov_b = float(np.mean(dist_gt_b <= args.coverage_radius)) if len(dist_gt_b) else float("nan")

    if args.random_sift_count < 0:
        raise ValueError("--random-sift-count must be >= 0.")
    random_sift_count_a = min(int(args.random_sift_count), int(len(kpts_a)))
    if random_sift_count_a > 0:
        rng_a = np.random.default_rng(args.random_sift_seed)
        sel_a = np.sort(rng_a.choice(len(kpts_a), size=random_sift_count_a, replace=False))
        sift_a_sel = kpts_a[sel_a]
        sift_a_sel_t = torch.from_numpy(sift_a_sel)[None].float()
        d_sift, valid_sift = sample_depth(sift_a_sel_t, depth_a_t)
        d_sift_used = d_sift * float(scale_a)
        uv_sift_b_t, vis_sift_b_t = project(
            sift_a_sel_t, d_sift_used, depth_b_t, cam_a, cam_b, T_a2b, valid_sift, ccth=None
        )
        uv_sift_b = uv_sift_b_t[0].detach().cpu().numpy()
        vis_sift_b = vis_sift_b_t[0].detach().cpu().numpy().astype(bool)
        sift_a_vis = sift_a_sel[vis_sift_b]
        sift_b_proj_vis = uv_sift_b[vis_sift_b]
    else:
        sift_a_sel = np.zeros((0, 2), dtype=np.float32)
        uv_sift_b = np.zeros((0, 2), dtype=np.float32)
        vis_sift_b = np.zeros((0,), dtype=bool)
        sift_a_vis = np.zeros((0, 2), dtype=np.float32)
        sift_b_proj_vis = np.zeros((0, 2), dtype=np.float32)

    random_sift_count_b = min(int(args.random_sift_count), int(len(kpts_b)))
    if random_sift_count_b > 0:
        rng_b = np.random.default_rng(args.random_sift_seed + 1)
        sel_b = np.sort(rng_b.choice(len(kpts_b), size=random_sift_count_b, replace=False))
        sift_b_sel = kpts_b[sel_b]
        sift_b_sel_t = torch.from_numpy(sift_b_sel)[None].float()
        d_sift_b, valid_sift_b = sample_depth(sift_b_sel_t, depth_b_t)
        d_sift_b_used = d_sift_b * float(scale_b)
        uv_sift_a_t, vis_sift_a_t = project(
            sift_b_sel_t, d_sift_b_used, depth_a_t, cam_b, cam_a, T_b2a, valid_sift_b, ccth=None
        )
        uv_sift_a = uv_sift_a_t[0].detach().cpu().numpy()
        vis_sift_a = vis_sift_a_t[0].detach().cpu().numpy().astype(bool)
        sift_b_vis = sift_b_sel[vis_sift_a]
        sift_a_proj_vis = uv_sift_a[vis_sift_a]
    else:
        sift_b_sel = np.zeros((0, 2), dtype=np.float32)
        uv_sift_a = np.zeros((0, 2), dtype=np.float32)
        vis_sift_a = np.zeros((0,), dtype=bool)
        sift_b_vis = np.zeros((0, 2), dtype=np.float32)
        sift_a_proj_vis = np.zeros((0, 2), dtype=np.float32)

    img_a_bgr = _to_bgr_image(img_a)
    img_b_bgr = _to_bgr_image(img_b)

    out_dir = args.out_dir / args.scene / f"{Path(name_a).stem}_{Path(name_b).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rainbow = _draw_rainbow_matches(img_a_bgr, img_b_bgr, pts_a, pts_b, args.max_draw)
    cv2.imwrite(str(out_dir / "gt_rainbow_matches.png"), rainbow)

    cv2.imwrite(str(out_dir / "cropped_viewA.png"), img_a_bgr)
    cv2.imwrite(str(out_dir / "cropped_viewB.png"), img_b_bgr)

    depth_a_vis = _depth_to_colorbar_image(view_a["depth"])
    depth_b_vis = _depth_to_colorbar_image(view_b["depth"])
    cv2.imwrite(str(out_dir / "depth_viewA.png"), depth_a_vis)
    cv2.imwrite(str(out_dir / "depth_viewB.png"), depth_b_vis)

    depth_overlay_a = _depth_overlay_image(img_a_bgr, view_a["depth"])
    depth_overlay_b = _depth_overlay_image(img_b_bgr, view_b["depth"])
    cv2.imwrite(str(out_dir / "depth_overlay_viewA.png"), depth_overlay_a)
    cv2.imwrite(str(out_dir / "depth_overlay_viewB.png"), depth_overlay_b)

    sift_reproj_pair = _draw_rainbow_matches(
        img_a_bgr, img_b_bgr, sift_a_vis, sift_b_proj_vis, args.max_draw
    )
    cv2.imwrite(str(out_dir / "sift_reprojection_A_to_B.png"), sift_reproj_pair)
    sift_reproj_pair_ba = _draw_rainbow_matches(
        img_b_bgr, img_a_bgr, sift_b_vis, sift_a_proj_vis, args.max_draw
    )
    cv2.imwrite(str(out_dir / "sift_reprojection_B_to_A.png"), sift_reproj_pair_ba)

    res_ab = _draw_residual_overlay(img_b_bgr, pts_b[vis_b], uv_b_proj[vis_b], err_ab[vis_b], args.max_draw)
    cv2.imwrite(str(out_dir / "residual_A_to_B.png"), res_ab)
    res_ba = _draw_residual_overlay(img_a_bgr, pts_a[vis_a], uv_a_proj[vis_a], err_ba[vis_a], args.max_draw)
    cv2.imwrite(str(out_dir / "residual_B_to_A.png"), res_ba)

    cov_img_a = _draw_coverage_overlay(img_a_bgr, pts_a, kpts_a, args.coverage_radius)
    cov_img_b = _draw_coverage_overlay(img_b_bgr, pts_b, kpts_b, args.coverage_radius)
    cv2.imwrite(str(out_dir / "coverage_viewA.png"), cov_img_a)
    cv2.imwrite(str(out_dir / "coverage_viewB.png"), cov_img_b)

    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("# Endomapper Dense projection inspection\n")
        f.write(f"scene={args.scene}\n")
        f.write(f"image_a={name_a}\n")
        f.write(f"image_b={name_b}\n")
        f.write(f"depth_scale_mode={scale_mode}\n")
        f.write("depth_scale_policy=always_colmap_median_ratio\n")
        f.write(f"shared_ids_raw={len(shared_ids_raw)}\n")
        f.write(f"shared_ids_after_crop={len(ids_crop)}\n")
        f.write(f"shared_ids_used_for_projection={len(pts_a)}\n")
        f.write(f"sampled_depth_valid_A={valid_a_count}\n")
        f.write(f"sampled_depth_valid_B={valid_b_count}\n")
        f.write(f"pose_dt={pose_dt}\n")
        f.write(f"pose_dr_deg={pose_dr}\n")
        f.write(f"depth_scale_A={scale_a}\n")
        f.write(f"depth_scale_B={scale_b}\n")
        f.write(f"depth_scale_A_median_ratio={scale_a_med}\n")
        f.write(f"depth_scale_B_median_ratio={scale_b_med}\n")
        f.write(f"depth_sample_A_raw_stats={depth_stats_a_raw}\n")
        f.write(f"depth_sample_B_raw_stats={depth_stats_b_raw}\n")
        f.write(f"depth_sample_A_aligned_stats={depth_stats_a_aligned}\n")
        f.write(f"depth_sample_B_aligned_stats={depth_stats_b_aligned}\n")
        f.write(f"colmap_z_A_stats={z_stats_a}\n")
        f.write(f"colmap_z_B_stats={z_stats_b}\n")
        f.write(f"scale_ratio_A_stats={ratio_stats_a}\n")
        f.write(f"scale_ratio_B_stats={ratio_stats_b}\n")
        f.write(f"scale_alignment_final_A={scale_a}\n")
        f.write(f"scale_alignment_final_B={scale_b}\n")
        f.write(f"visible_A={int(np.sum(vis_a))}\n")
        f.write(f"visible_B={int(np.sum(vis_b))}\n")
        f.write(f"stats_A_to_B={stats_ab}\n")
        f.write(f"stats_B_to_A={stats_ba}\n")
        f.write(f"inlier_A_to_B={inliers_ab}\n")
        f.write(f"inlier_B_to_A={inliers_ba}\n")
        f.write(f"cudasift_kpts_A={len(kpts_a)}\n")
        f.write(f"cudasift_kpts_B={len(kpts_b)}\n")
        f.write(f"coverage_radius_px={args.coverage_radius}\n")
        f.write(f"coverage_A={cov_a}\n")
        f.write(f"coverage_B={cov_b}\n")
        f.write(f"random_sift_count_requested={int(args.random_sift_count)}\n")
        f.write(f"random_sift_count_selected_A={random_sift_count_a}\n")
        f.write(f"random_sift_count_selected_B={random_sift_count_b}\n")
        f.write(f"random_sift_seed={int(args.random_sift_seed)}\n")
        f.write(f"random_sift_visible_in_B={int(np.sum(vis_sift_b))}\n")
        f.write(f"random_sift_visible_in_A={int(np.sum(vis_sift_a))}\n")

    print(f"[ok] scene={args.scene}, pair=({name_a}, {name_b})")
    print(f"[ok] shared={len(pts_a)} visible_B={int(np.sum(vis_b))} visible_A={int(np.sum(vis_a))}")
    print(f"[ok] sampled_depth_valid A={valid_a_count} B={valid_b_count} pose_dt={pose_dt:.6f}m")
    print(
        f"[ok] depth scales applied ({scale_mode}): "
        f"A={float(scale_a):.4f}, B={float(scale_b):.4f}"
    )
    print(f"[ok] inlier@3px A->B={inliers_ab[3.0]:.4f} B->A={inliers_ba[3.0]:.4f}")
    print(f"[ok] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
