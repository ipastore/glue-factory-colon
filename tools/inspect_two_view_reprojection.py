#!/usr/bin/env python3
"""Two-view projection inspection for Endomapper (training-cache NPZ).

This mimics the training-time workflow by loading a preprocessed Endomapper
`.npz` cache and running `gt_matches_from_pose_sparse_map`, which exercises
`gluefactory/geometry/depth.py:project()` (including distortion paths).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import cv2

from gluefactory.geometry.gt_generation import gt_matches_from_pose_sparse_map
from gluefactory.geometry.wrappers import Camera, Pose


def _resolve_image_index(image_names: np.ndarray, token: str) -> int:
    idx = np.where(image_names == token)[0]
    if idx.size == 0:
        raise ValueError(f"Unknown image name '{token}' (not found in npz image_names).")
    return int(idx[0])


def _obs_by_point_id(keypoints: np.ndarray, point3d_ids: np.ndarray) -> Dict[int, np.ndarray]:
    obs: Dict[int, np.ndarray] = {}
    for xy, pid in zip(keypoints, point3d_ids):
        if pid == -1:
            continue
        if pid not in obs:
            obs[int(pid)] = np.asarray(xy, dtype=np.float32)
    return obs


def _percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _summarize(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(errors.size),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p90": _percentile(errors, 90),
        "p95": _percentile(errors, 95),
        "max": float(errors.max()),
    }


def _load_image(seq_root: Path, map_id: str, name: str) -> Optional[np.ndarray]:
    img_path = seq_root / "output" / "3D_maps" / map_id / "keyframes" / f"Keyframe_{name}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    return None


def _draw_overlay(
    canvas: np.ndarray,
    measured_uv: np.ndarray,
    projected_uv: np.ndarray,
    errors: np.ndarray,
    *,
    max_draw: int = 2000,
    radius: int = 2,
) -> np.ndarray:


    if measured_uv.size == 0:
        return canvas

    if measured_uv.shape != projected_uv.shape:
        raise ValueError("Measured/projected UV shapes do not match.")

    idx = np.arange(measured_uv.shape[0])
    if max_draw > 0 and idx.size > max_draw:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_draw, replace=False)

    emax = float(np.max(errors[idx])) if idx.size else 1.0
    emax = max(emax, 1e-6)

    out = canvas.copy()
    for i in idx:
        u_m, v_m = measured_uv[i]
        u_p, v_p = projected_uv[i]
        e = float(errors[i])
        t = min(max(e / emax, 0.0), 1.0)
        # Only use blue (measured) and green (reprojected/residual) for clarity.
        color = (0, 255, 0)  # BGR: green
        cv2.line(
            out,
            (int(round(u_p)), int(round(v_p))),
            (int(round(u_m)), int(round(v_m))),
            color,
            1,
            cv2.LINE_AA,
        )
        # Draw reprojected first (green), then measured on top (blue) for visibility.
        cv2.circle(out, (int(round(u_p)), int(round(v_p))), radius, (0, 255, 0), -1)
        cv2.circle(out, (int(round(u_m)), int(round(v_m))), radius, (255, 0, 0), -1)
    return out


def _draw_points(
    canvas: np.ndarray,
    uv: np.ndarray,
    *,
    max_draw: int = 2000,
    radius: int = 2,
    color: Tuple[int, int, int] = (255, 0, 0),  # BGR
) -> np.ndarray:

    if uv.size == 0:
        return canvas

    idx = np.arange(uv.shape[0])
    if max_draw > 0 and idx.size > max_draw:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_draw, replace=False)

    out = canvas.copy()
    for i in idx:
        u, v = uv[i]
        cv2.circle(out, (int(round(u)), int(round(v))), radius, color, -1)
    return out


def _draw_two_color_points(
    canvas: np.ndarray,
    uv_blue: np.ndarray,
    uv_green: np.ndarray,
    *,
    max_draw: int = 2000,
    radius: int = 2,
) -> np.ndarray:
    out = canvas.copy()
    out = _draw_points(out, uv_green, max_draw=max_draw, radius=radius, color=(0, 255, 0))
    out = _draw_points(out, uv_blue, max_draw=max_draw, radius=radius, color=(255, 0, 0))
    return out


def _draw_labels(
    canvas: np.ndarray,
    uv: np.ndarray,
    labels: np.ndarray,
    *,
    color: Tuple[int, int, int] = (255, 255, 255),  # BGR
    offset: Tuple[int, int] = (4, -4),
) -> np.ndarray:
    if uv.size == 0:
        return canvas

    if uv.shape[0] != labels.shape[0]:
        raise ValueError("UV/labels shapes do not match.")

    out = canvas.copy()
    for (u, v), lab in zip(uv, labels):
        x = int(round(u + offset[0]))
        y = int(round(v + offset[1]))
        text = str(int(lab))
        cv2.putText(
            out,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _hstack_pad(left: np.ndarray, right: np.ndarray, *, pad_value: int = 255) -> np.ndarray:
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("Expected color images (H, W, 3).")
    if left.shape[2] != 3 or right.shape[2] != 3:
        raise ValueError("Expected 3-channel images.")

    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    out = np.full((h, w, 3), pad_value, dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] :] = right
    return out


def _camera_from_npz(npz: Dict, idx: int) -> Camera:
    K = npz["intrinsics"][idx].astype(np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    width, height = (int(x) for x in npz["image_sizes"][idx].tolist())
    model = str(np.asarray(npz["camera_model"]).item())
    dist = npz["distortion_coeffs"][idx].astype(np.float32)
    if model == "OPENCV_FISHEYE":
        params = np.array([fx, fy, cx, cy, *dist.tolist()], dtype=np.float32)
    elif model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        params = np.array([fx, fy, cx, cy], dtype=np.float32)
    else:
        # Fallback: keep intrinsics and no distortion.
        params = np.array([fx, fy, cx, cy], dtype=np.float32)
        model = "PINHOLE"
    cam = Camera.from_colmap({"model": model, "width": width, "height": height, "params": params})
    return cam.float()


def _lookup_point3d_xyz(npz: Dict, point3d_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lookup 3D point coordinates for given point ids.

    Returns:
        xyz: (N,3) float32 (undefined for missing ids)
        ok: (N,) bool whether the id exists in the map
    """
    ids_all = npz["point3D_ids"].astype(np.int64)
    xyz_all = npz["point3D_coords"].astype(np.float32)
    order = np.argsort(ids_all)
    ids_sorted = ids_all[order]
    xyz_sorted = xyz_all[order]

    q = point3d_ids.astype(np.int64)
    pos = np.searchsorted(ids_sorted, q)
    ok = (pos >= 0) & (pos < ids_sorted.shape[0]) & (ids_sorted[pos] == q)
    xyz = np.zeros((q.shape[0], 3), dtype=np.float32)
    if ok.any():
        xyz[ok] = xyz_sorted[pos[ok]]
    return xyz, ok


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("data/Endomapper_CUDASIFT/processed_npz/Seq_024_a_map1.npz"),
        help="Processed Endomapper cache NPZ (same format as training).",
    )
    parser.add_argument(
        "--seq-root",
        type=Path,
        default=None,
        help="Optional sequence folder to load keyframe images for overlays.",
    )
    parser.add_argument(
        "--map-id",
        type=str,
        default=None,
        help="Optional map id for image loading (defaults to npz map_id).",
    )
    parser.add_argument(
        "--a",
        required=True,
        help="First image name (images.txt `name` field).",
    )
    parser.add_argument(
        "--b",
        required=True,
        help="Second image name (images.txt `name` field).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/two_view_reprojection"),
        help="Directory to write overlays/metrics.",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=0,
        help="Max correspondences to draw (0 = draw all).",
    )
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw numeric labels for a few correspondences.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Also compute oracle reprojection using sparse-map 3D points.",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    npz = np.load(str(args.npz), allow_pickle=True)
    image_names = npz["image_names"]
    idx_a = _resolve_image_index(image_names, args.a)
    idx_b = _resolve_image_index(image_names, args.b)

    cam_a = _camera_from_npz(npz, idx_a)
    cam_b = _camera_from_npz(npz, idx_b)

    T_w2a = Pose.from_4x4mat(torch.from_numpy(npz["poses"][idx_a]).float())
    T_w2b = Pose.from_4x4mat(torch.from_numpy(npz["poses"][idx_b]).float())
    T_a2b = T_w2b.compose(T_w2a.inv())
    T_b2a = T_a2b.inv()

    kp0_np = npz["keypoints_per_image"][idx_a].astype(np.float32)
    kp1_np = npz["keypoints_per_image"][idx_b].astype(np.float32)
    ids0_np = npz["point3D_ids_per_image"][idx_a].astype(np.int64)
    ids1_np = npz["point3D_ids_per_image"][idx_b].astype(np.int64)
    d0_np = npz["depths_per_image"][idx_a].astype(np.float32)
    d1_np = npz["depths_per_image"][idx_b].astype(np.float32)
    valid_d0_np = npz["valid_depth_mask_per_image"][idx_a].astype(bool)
    valid_d1_np = npz["valid_depth_mask_per_image"][idx_b].astype(bool)
    valid_3d0_np = npz["valid_3d_mask_per_image"][idx_a].astype(bool)
    valid_3d1_np = npz["valid_3d_mask_per_image"][idx_b].astype(bool)

    kp0 = torch.from_numpy(kp0_np[None]).float()
    kp1 = torch.from_numpy(kp1_np[None]).float()
    ids0 = torch.from_numpy(ids0_np[None])
    ids1 = torch.from_numpy(ids1_np[None])
    d0 = torch.from_numpy(d0_np[None]).float()
    d1 = torch.from_numpy(d1_np[None]).float()
    valid_d0 = torch.from_numpy(valid_d0_np[None]).bool()
    valid_d1 = torch.from_numpy(valid_d1_np[None]).bool()
    valid_3d0 = torch.from_numpy(valid_3d0_np[None]).bool()
    valid_3d1 = torch.from_numpy(valid_3d1_np[None]).bool()

    data = {
        "view0": {"camera": cam_a},
        "view1": {"camera": cam_b},
        "T_0to1": T_a2b,
        "T_1to0": T_b2a,
    }

    gt = gt_matches_from_pose_sparse_map(
        kp0,
        kp1,
        data,
        pos_th=None,
        neg_th=None,
        point3D_ids0=ids0,
        point3D_ids1=ids1,
        valid_3D_mask0=valid_3d0,
        valid_3D_mask1=valid_3d1,
        sparse_depth0=d0,
        sparse_depth1=d1,
        valid_depth_mask0=valid_d0,
        valid_depth_mask1=valid_d1,
    )

    proj_0to1 = gt["proj_0to1"][0].cpu().numpy()
    proj_1to0 = gt["proj_1to0"][0].cpu().numpy()
    visible0 = gt["visible0"][0].cpu().numpy().astype(bool)
    visible1 = gt["visible1"][0].cpu().numpy().astype(bool)

    obs_a = _obs_by_point_id(kp0_np, ids0_np)
    obs_b = _obs_by_point_id(kp1_np, ids1_np)

    idx0 = np.where(valid_d0_np & valid_3d0_np & (ids0_np != -1))[0]
    pid0 = ids0_np[idx0].astype(np.int64)
    has_in_b = np.array([int(pid) in obs_b for pid in pid0], dtype=bool)
    idx0 = idx0[has_in_b]
    if idx0.size:
        uv_b_meas = np.stack([obs_b[int(pid)] for pid in ids0_np[idx0]], axis=0)
        uv_b_proj = proj_0to1[idx0]
        vis0 = visible0[idx0]
        err_ab = np.linalg.norm(uv_b_proj - uv_b_meas, axis=-1)
        err_ab_v = err_ab[vis0]
    else:
        uv_b_meas = np.zeros((0, 2), dtype=np.float32)
        uv_b_proj = np.zeros((0, 2), dtype=np.float32)
        vis0 = np.zeros((0,), dtype=bool)
        err_ab = np.zeros((0,), dtype=np.float32)
        err_ab_v = err_ab

    idx1 = np.where(valid_d1_np & valid_3d1_np & (ids1_np != -1))[0]
    pid1 = ids1_np[idx1].astype(np.int64)
    has_in_a = np.array([int(pid) in obs_a for pid in pid1], dtype=bool)
    idx1 = idx1[has_in_a]
    if idx1.size:
        uv_a_meas = np.stack([obs_a[int(pid)] for pid in ids1_np[idx1]], axis=0)
        uv_a_proj = proj_1to0[idx1]
        vis1 = visible1[idx1]
        err_ba = np.linalg.norm(uv_a_proj - uv_a_meas, axis=-1)
        err_ba_v = err_ba[vis1]
    else:
        uv_a_meas = np.zeros((0, 2), dtype=np.float32)
        uv_a_proj = np.zeros((0, 2), dtype=np.float32)
        vis1 = np.zeros((0,), dtype=bool)
        err_ba = np.zeros((0,), dtype=np.float32)
        err_ba_v = err_ba

    print(
        f"A->B tracks: {idx0.size} | visible: {int(vis0.sum())} "
        f"({(vis0.mean()*100 if vis0.size else 0):.1f}%)"
    )
    print("A->B reprojection error (px):", _summarize(err_ab_v))
    print(
        f"B->A tracks: {idx1.size} | visible: {int(vis1.sum())} "
        f"({(vis1.mean()*100 if vis1.size else 0):.1f}%)"
    )
    print("B->A reprojection error (px):", _summarize(err_ba_v))
    inlier_rates = {}
    for thr in (1.0, 2.0, 3.0, 4.0 ,5.0):
        iab = float(np.mean(err_ab_v <= thr)) if err_ab_v.size else float("nan")
        iba = float(np.mean(err_ba_v <= thr)) if err_ba_v.size else float("nan")
        inlier_rates[thr] = (iab, iba)
        print(f"inlier@{thr:g}px: A->B={iab:.3f} B->A={iba:.3f}")

    err_ab_oracle_v = np.zeros((0,), dtype=np.float32)
    err_ba_oracle_v = np.zeros((0,), dtype=np.float32)
    if args.oracle:
        if idx0.size:
            xyz_w0, ok0 = _lookup_point3d_xyz(npz, ids0_np[idx0])
            xyz_w0_t = torch.from_numpy(xyz_w0[ok0]).float()
            uv_b_oracle_t, v_oracle_t = cam_b.cam2image(T_w2b.transform(xyz_w0_t))
            uv_b_oracle = uv_b_oracle_t.cpu().numpy()
            v_oracle = v_oracle_t.cpu().numpy().astype(bool)
            uv_b_meas_ok = uv_b_meas[ok0]
            err_ab_oracle = np.linalg.norm(uv_b_oracle - uv_b_meas_ok, axis=-1)
            err_ab_oracle_v = err_ab_oracle[v_oracle]
        if idx1.size:
            xyz_w1, ok1 = _lookup_point3d_xyz(npz, ids1_np[idx1])
            xyz_w1_t = torch.from_numpy(xyz_w1[ok1]).float()
            uv_a_oracle_t, v_oracle_t = cam_a.cam2image(T_w2a.transform(xyz_w1_t))
            uv_a_oracle = uv_a_oracle_t.cpu().numpy()
            v_oracle = v_oracle_t.cpu().numpy().astype(bool)
            uv_a_meas_ok = uv_a_meas[ok1]
            err_ba_oracle = np.linalg.norm(uv_a_oracle - uv_a_meas_ok, axis=-1)
            err_ba_oracle_v = err_ba_oracle[v_oracle]
        print("A->B oracle reprojection error (px):", _summarize(err_ab_oracle_v))
        print("B->A oracle reprojection error (px):", _summarize(err_ba_oracle_v))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name_a = str(image_names[idx_a])
    name_b = str(image_names[idx_b])
    map_id = str(args.map_id) if args.map_id is not None else str(np.asarray(npz["map_id"]).item())
    residuals_path = args.out_dir / f"projection_metrics_map{map_id}_a{name_a}_b{name_b}.txt"
    stats_ab = _summarize(err_ab_v)
    stats_ba = _summarize(err_ba_v)
    stats_ab_oracle = _summarize(err_ab_oracle_v)
    stats_ba_oracle = _summarize(err_ba_oracle_v)
    with residuals_path.open("w", encoding="utf-8") as f:
        f.write("# Two-view projection metrics (gt_matches_from_pose_sparse_map / depth.project)\n")
        f.write(f"npz {args.npz}\n")
        f.write(f"map_id {map_id}\n")
        f.write(f"image_a_name {name_a}\n")
        f.write(f"image_b_name {name_b}\n")
        f.write(f"camera_a_model {cam_a.model}\n")
        f.write(f"camera_b_model {cam_b.model}\n")
        f.write(f"a_to_b_tracks {int(idx0.size)}\n")
        f.write(f"a_to_b_visible {int(vis0.sum())}\n")
        f.write(f"b_to_a_tracks {int(idx1.size)}\n")
        f.write(f"b_to_a_visible {int(vis1.sum())}\n")

        f.write("a_to_b_count %d\n" % stats_ab["count"])
        f.write("a_to_b_mean_px %.6f\n" % stats_ab["mean"])
        f.write("a_to_b_median_px %.6f\n" % stats_ab["median"])
        f.write("a_to_b_p90_px %.6f\n" % stats_ab["p90"])
        f.write("a_to_b_p95_px %.6f\n" % stats_ab["p95"])
        f.write("a_to_b_max_px %.6f\n" % stats_ab["max"])

        f.write("b_to_a_count %d\n" % stats_ba["count"])
        f.write("b_to_a_mean_px %.6f\n" % stats_ba["mean"])
        f.write("b_to_a_median_px %.6f\n" % stats_ba["median"])
        f.write("b_to_a_p90_px %.6f\n" % stats_ba["p90"])
        f.write("b_to_a_p95_px %.6f\n" % stats_ba["p95"])
        f.write("b_to_a_max_px %.6f\n" % stats_ba["max"])

        if args.oracle:
            f.write("a_to_b_oracle_count %d\n" % stats_ab_oracle["count"])
            f.write("a_to_b_oracle_mean_px %.6f\n" % stats_ab_oracle["mean"])
            f.write("a_to_b_oracle_median_px %.6f\n" % stats_ab_oracle["median"])
            f.write("a_to_b_oracle_p90_px %.6f\n" % stats_ab_oracle["p90"])
            f.write("a_to_b_oracle_p95_px %.6f\n" % stats_ab_oracle["p95"])
            f.write("a_to_b_oracle_max_px %.6f\n" % stats_ab_oracle["max"])

            f.write("b_to_a_oracle_count %d\n" % stats_ba_oracle["count"])
            f.write("b_to_a_oracle_mean_px %.6f\n" % stats_ba_oracle["mean"])
            f.write("b_to_a_oracle_median_px %.6f\n" % stats_ba_oracle["median"])
            f.write("b_to_a_oracle_p90_px %.6f\n" % stats_ba_oracle["p90"])
            f.write("b_to_a_oracle_p95_px %.6f\n" % stats_ba_oracle["p95"])
            f.write("b_to_a_oracle_max_px %.6f\n" % stats_ba_oracle["max"])

        for thr, (ia, ib) in inlier_rates.items():
            key = str(int(thr)) if float(thr).is_integer() else str(thr)
            f.write(f"inlier_ab_at_{key}px {ia:.6f}\n")
            f.write(f"inlier_ba_at_{key}px {ib:.6f}\n")
    print("Wrote", residuals_path)

    if args.no_viz:
        return 0

    seq_name = str(np.asarray(npz["seq"]).item())
    root_guess = args.npz.parent.parent
    seq_root = args.seq_root if args.seq_root is not None else (root_guess / seq_name)
    base_a = _load_image(seq_root, map_id, name_a)
    if base_a is None:
        base_a = np.full(
            (int(npz["image_sizes"][idx_a][1]), int(npz["image_sizes"][idx_a][0]), 3),
            255,
            np.uint8,
        )
        cv2.putText(
            base_a,
            f"no image found for '{name_a}'",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    base_b = _load_image(seq_root, map_id, name_b)
    if base_b is None:
        base_b = np.full(
            (int(npz["image_sizes"][idx_b][1]), int(npz["image_sizes"][idx_b][0]), 3),
            255,
            np.uint8,
        )
        cv2.putText(
            base_b,
            f"no image found for '{name_b}'",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # A -> B overlay: left shows A keypoints (blue), right shows B measured (blue) and projected (green).
    overlay_a_meas = _draw_points(
        base_a,
        kp0_np,
        max_draw=args.max_draw,
        color=(255, 0, 0),  # blue
    )
    overlay_b_ab = _draw_overlay(
        base_b,
        uv_b_meas[vis0],
        uv_b_proj[vis0],
        err_ab[vis0],
        max_draw=args.max_draw,
    )

    # B -> A overlay: left shows A measured (blue) and projected (green), right shows B keypoints (blue).
    overlay_a_ba = _draw_overlay(
        base_a,
        uv_a_meas[vis1],
        uv_a_proj[vis1],
        err_ba[vis1],
        max_draw=args.max_draw,
    )
    overlay_b_meas = _draw_points(
        base_b,
        kp1_np,
        max_draw=args.max_draw,
        color=(255, 0, 0),  # blue
    )

    if args.draw_labels:
        label_max = 10
        idx = np.arange(min(label_max, uv_b_meas.shape[0]))
        if idx.size > label_max:
            idx = idx[:label_max]
        if idx.size:
            labels = np.arange(idx.size, dtype=np.int64)
            # A -> B: label A measured and B projected.
            overlay_a_meas = _draw_labels(
                overlay_a_meas,
                uv_a_meas[idx],
                labels,
                color=(255, 0, 0),
            )
            overlay_b_ab = _draw_labels(
                overlay_b_ab,
                uv_b_proj[idx],
                labels,
                color=(255, 0, 0),
            )
            # B -> A: label B measured and A projected.
            overlay_b_meas = _draw_labels(
                overlay_b_meas,
                uv_b_meas[idx],
                labels,
                color=(255, 0, 0),
            )
            overlay_a_ba = _draw_labels(
                overlay_a_ba,
                uv_a_proj[idx],
                labels,
                color=(255, 0, 0),
            )

    out_ab = args.out_dir / f"overlay_ab_map{map_id}_a{name_a}_b{name_b}.png"
    out_ba = args.out_dir / f"overlay_ba_map{map_id}_a{name_a}_b{name_b}.png"

    cv2.imwrite(str(out_ab), _hstack_pad(overlay_a_meas, overlay_b_ab))
    cv2.imwrite(str(out_ba), _hstack_pad(overlay_a_ba, overlay_b_meas))
    print("Wrote", out_ab)
    print("Wrote", out_ba)

    # Round-trip camera check on image B: unproject->project->unproject->project.
    with torch.no_grad():
        uv0 = torch.from_numpy(kp1_np).double()
        in_img = cam_b.in_image(uv0[None]).squeeze(0)
        uv0 = uv0[in_img]
        rays1 = cam_b.image2cam(uv0)
        uv1, v1 = cam_b.cam2image(rays1)
        rays2 = cam_b.image2cam(uv1)
        uv2, v2 = cam_b.cam2image(rays2)
        v = (v1 & v2).cpu().numpy().astype(bool)
        uv0_np = uv0.cpu().numpy()
        uv2_np = uv2.cpu().numpy()
        err_rt = np.linalg.norm(uv2_np - uv0_np, axis=-1)
        err_rt_v = err_rt[v]
    print("B roundtrip reprojection error (px):", _summarize(err_rt_v))

    overlay_rt_b = _draw_two_color_points(base_b, uv0_np[v], uv2_np[v], max_draw=args.max_draw)
    out_rt_b = args.out_dir / f"overlay_roundtrip_b_map{map_id}_b{name_b}.png"
    cv2.imwrite(str(out_rt_b), overlay_rt_b)
    print("Wrote", out_rt_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
