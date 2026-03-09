#!/usr/bin/env python3
"""Inspect EndomapperDense reprojection using cached depth_keypoints only."""

from __future__ import annotations

import argparse
import colorsys
import re
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from gluefactory.datasets.endomapper_dense import _PairDataset
from gluefactory.datasets.endomapper_utils import read_images_txt
from gluefactory.geometry.depth import project
from gluefactory.settings import DATA_PATH
from gluefactory.utils.image import ImagePreprocessor, load_image


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect EndomapperDense projections with cached depth_keypoints only "
            "(no runtime depth sampling / no scale alignment)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_PATH / "endomapper_dense",
        help="Root folder of endomapper_dense dataset.",
    )
    parser.add_argument(
        "--cache-path-template",
        type=str,
        required=True,
        help=(
            "CacheLoader path template relative to DATA_PATH, e.g. "
            "'exports/endomapper-dense-undist-depth-r1024_py-cudasift-k2048/{seq_map}.h5'"
        ),
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene token, e.g. Seq_008_a_map0.",
    )
    parser.add_argument(
        "--image-a",
        required=True,
        help="First image token (exact name, basename, or stem).",
    )
    parser.add_argument(
        "--image-b",
        required=True,
        help="Second image token (exact name, basename, or stem).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/endomapper_dense_projection_cached"),
        help="Output directory for overlays and metrics.",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=2000,
        help="Maximum correspondences to draw (0 means all).",
    )
    parser.add_argument(
        "--depth-assign-radius",
        type=float,
        default=3.0,
        help=(
            "Maximum distance (px) from GT point to cached keypoint to assign "
            "a cached depth_keypoint."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_scene(scene: str) -> tuple[str, str]:
    match = re.match(r"^(Seq_.*)_map(\d+)$", scene)
    if not match:
        raise ValueError(
            f"Invalid scene '{scene}'. Expected format: Seq_XXX_Y_mapN."
        )
    return match.group(1), match.group(2)


def _obs_by_id(colmap_image):
    out = {}
    for xy, pid in zip(colmap_image.xys, colmap_image.point3D_ids):
        if pid == -1:
            continue
        pid_i = int(pid)
        if pid_i not in out:
            out[pid_i] = np.asarray(xy, dtype=np.float32)
    return out


def _crop_points(uv: np.ndarray, left_top: tuple[float, float], w: int, h: int):
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


def _make_loader(data_root: Path, scene: str, cache_path_template: str):
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
            "load_features": {
                "do": True,
                "path": cache_path_template,
                "add_data_path": True,
                "collate": False,
                "data_keys": ["keypoints", "depth_keypoints", "valid_depth_keypoints"],
            },
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


def _draw_rainbow_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    max_draw: int,
) -> np.ndarray:
    h = max(img_a.shape[0], img_b.shape[0])
    w = img_a.shape[1] + img_b.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
    canvas[: img_b.shape[0], img_a.shape[1] :] = img_b

    idx = _sample_indices(len(pts_a), max_draw)
    n = max(1, len(idx))
    shift = img_a.shape[1]
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


def _draw_points(img: np.ndarray, pts: np.ndarray, color=(0, 255, 0), radius=2):
    out = img.copy()
    for p in pts:
        cv2.circle(
            out,
            (int(round(float(p[0]))), int(round(float(p[1])))),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )
    return out


def _nearest_points(queries: np.ndarray, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(queries) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    if len(refs) == 0:
        return (
            np.zeros((len(queries), 2), dtype=np.float32),
            np.full((len(queries),), np.inf, dtype=np.float32),
        )
    best_d2 = np.full((len(queries),), np.inf, dtype=np.float32)
    best_idx = np.zeros((len(queries),), dtype=np.int64)
    block = 4096
    for i in range(0, len(refs), block):
        r = refs[i : i + block]
        d = queries[:, None, :] - r[None, :, :]
        d2 = np.sum(d * d, axis=-1)
        loc = np.argmin(d2, axis=1)
        val = d2[np.arange(len(queries)), loc]
        better = val < best_d2
        if np.any(better):
            best_d2[better] = val[better]
            best_idx[better] = i + loc[better]
    return refs[best_idx].astype(np.float32, copy=False), np.sqrt(best_d2)


def _assign_cached_depth_to_gt(
    gt_pts: np.ndarray,
    cached_kpts: np.ndarray,
    cached_depth: np.ndarray,
    cached_valid: np.ndarray,
    max_dist_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest, dist = _nearest_points(gt_pts, cached_kpts)
    if len(gt_pts) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float32),
        )
    # map nearest coordinates back to index
    d = nearest[:, None, :] - cached_kpts[None, :, :]
    d2 = np.sum(d * d, axis=-1)
    nn_idx = np.argmin(d2, axis=1)
    depth = cached_depth[nn_idx].astype(np.float32, copy=False)
    valid = cached_valid[nn_idx] & np.isfinite(depth) & (depth > 0.0) & (dist <= max_dist_px)
    return depth, valid, dist


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


def _summary_stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    loader = _make_loader(args.data_root, args.scene, args.cache_path_template)
    seq, map_id = _parse_scene(args.scene)

    if args.scene not in loader.image_names:
        raise KeyError(f"Scene '{args.scene}' not found in loaded split.")

    image_names = loader.image_names[args.scene]
    idx_a = _resolve_image_index(image_names, args.image_a)
    idx_b = _resolve_image_index(image_names, args.image_b)
    name_a = str(image_names[idx_a])
    name_b = str(image_names[idx_b])

    if not loader.valid[args.scene][idx_a] or not loader.valid[args.scene][idx_b]:
        raise ValueError(
            "Selected pair is not valid for dataloader (missing image/depth). "
            f"idx_a={idx_a}, idx_b={idx_b}."
        )

    view_a = loader._read_view(args.scene, name_a)
    view_b = loader._read_view(args.scene, name_b)
    cache_a = view_a["cache"]
    cache_b = view_b["cache"]

    kpts_a = cache_a["keypoints"].detach().cpu().numpy().astype(np.float32, copy=False)
    kpts_b = cache_b["keypoints"].detach().cpu().numpy().astype(np.float32, copy=False)
    d_kp_a = cache_a["depth_keypoints"].detach().cpu().numpy().astype(np.float32, copy=False)
    d_kp_b = cache_b["depth_keypoints"].detach().cpu().numpy().astype(np.float32, copy=False)
    v_kp_a = cache_a["valid_depth_keypoints"].detach().cpu().numpy().astype(bool, copy=False)
    v_kp_b = cache_b["valid_depth_keypoints"].detach().cpu().numpy().astype(bool, copy=False)

    # Build GT matches from COLMAP shared point3D ids, then crop to dataloader view.
    colmap_dir = args.data_root / "Undistorted_SfM" / seq / map_id / "results_txt"
    images = read_images_txt(colmap_dir / "images.txt")
    by_name = {v.name: v for v in images.values()}
    if name_a not in by_name or name_b not in by_name:
        raise KeyError(f"Pair names not present in images.txt: '{name_a}', '{name_b}'.")
    obs_a = _obs_by_id(by_name[name_a])
    obs_b = _obs_by_id(by_name[name_b])
    shared_ids = sorted(set(obs_a.keys()) & set(obs_b.keys()))
    if len(shared_ids) == 0:
        raise ValueError(f"No shared point3D_ids between '{name_a}' and '{name_b}'.")
    gt_a_raw = np.stack([obs_a[pid] for pid in shared_ids], axis=0).astype(np.float32, copy=False)
    gt_b_raw = np.stack([obs_b[pid] for pid in shared_ids], axis=0).astype(np.float32, copy=False)

    scene_info_path = args.data_root / "scene_info" / f"{args.scene}.npz"
    with np.load(str(scene_info_path), allow_pickle=True) as scene_info:
        raw_img_a = load_image(args.data_root / str(scene_info["image_paths"][idx_a]), grayscale=False)
        raw_img_b = load_image(args.data_root / str(scene_info["image_paths"][idx_b]), grayscale=False)
    pre = ImagePreprocessor({})
    _, off_a = pre.crop_endomapper_dense(raw_img_a)
    _, off_b = pre.crop_endomapper_dense(raw_img_b)
    h, w = view_a["image"].shape[-2:]
    gt_a, in_a = _crop_points(gt_a_raw, off_a, w, h)
    gt_b, in_b = _crop_points(gt_b_raw, off_b, w, h)
    in_crop = in_a & in_b
    gt_a = gt_a[in_crop]
    gt_b = gt_b[in_crop]
    if len(gt_a) == 0:
        raise ValueError("No GT correspondences remain after cropping.")

    d_gt_a, valid_gt_a, nn_dist_a = _assign_cached_depth_to_gt(
        gt_a, kpts_a, d_kp_a, v_kp_a, args.depth_assign_radius
    )
    d_gt_b, valid_gt_b, nn_dist_b = _assign_cached_depth_to_gt(
        gt_b, kpts_b, d_kp_b, v_kp_b, args.depth_assign_radius
    )

    gt_a_t = torch.from_numpy(gt_a)[None].float()
    gt_b_t = torch.from_numpy(gt_b)[None].float()
    d_a_t = torch.from_numpy(d_gt_a)[None].float()
    d_b_t = torch.from_numpy(d_gt_b)[None].float()
    valid_a_t = torch.from_numpy(valid_gt_a)[None].bool()
    valid_b_t = torch.from_numpy(valid_gt_b)[None].bool()

    cam_a = view_a["camera"]
    cam_b = view_b["camera"]
    T_a2b = view_b["T_w2cam"] @ view_a["T_w2cam"].inv()
    T_b2a = view_a["T_w2cam"] @ view_b["T_w2cam"].inv()

    uv_b_proj_t, vis_ab_t = project(
        gt_a_t, d_a_t, None, cam_a, cam_b, T_a2b, valid_a_t, ccth=None
    )
    uv_a_proj_t, vis_ba_t = project(
        gt_b_t, d_b_t, None, cam_b, cam_a, T_b2a, valid_b_t, ccth=None
    )
    uv_b_proj = uv_b_proj_t[0].detach().cpu().numpy()
    uv_a_proj = uv_a_proj_t[0].detach().cpu().numpy()
    vis_ab = vis_ab_t[0].detach().cpu().numpy().astype(bool)
    vis_ba = vis_ba_t[0].detach().cpu().numpy().astype(bool)
    err_ab = np.linalg.norm(uv_b_proj - gt_b, axis=1)
    err_ba = np.linalg.norm(uv_a_proj - gt_a, axis=1)
    err_ab_v = err_ab[vis_ab]
    err_ba_v = err_ba[vis_ba]
    total_gt = int(len(gt_a))

    def _recover_percent(errors: np.ndarray, thr: float, denom: int) -> float:
        if denom == 0:
            return float("nan")
        if len(errors) == 0:
            return 0.0
        return float(100.0 * np.sum(errors <= thr) / denom)

    recover_ab = {t: _recover_percent(err_ab_v, t, total_gt) for t in [1, 2, 3, 4, 5]}
    recover_ba = {t: _recover_percent(err_ba_v, t, total_gt) for t in [1, 2, 3, 4, 5]}

    img_a_bgr = _to_bgr_image(view_a["image"])
    img_b_bgr = _to_bgr_image(view_b["image"])
    out_dir = args.out_dir / args.scene / f"{Path(name_a).stem}_{Path(name_b).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "cropped_viewA.png"), img_a_bgr)
    cv2.imwrite(str(out_dir / "cropped_viewB.png"), img_b_bgr)
    cv2.imwrite(str(out_dir / "cached_keypoints_viewA.png"), _draw_points(img_a_bgr, kpts_a))
    cv2.imwrite(str(out_dir / "cached_keypoints_viewB.png"), _draw_points(img_b_bgr, kpts_b))
    cv2.imwrite(
        str(out_dir / "cached_reprojection_A_to_B.png"),
        _draw_rainbow_matches(img_a_bgr, img_b_bgr, gt_a[vis_ab], uv_b_proj[vis_ab], args.max_draw),
    )
    cv2.imwrite(
        str(out_dir / "cached_reprojection_B_to_A.png"),
        _draw_rainbow_matches(img_b_bgr, img_a_bgr, gt_b[vis_ba], uv_a_proj[vis_ba], args.max_draw),
    )
    cv2.imwrite(
        str(out_dir / "gt_rainbow_matches.png"),
        _draw_rainbow_matches(img_a_bgr, img_b_bgr, gt_a, gt_b, args.max_draw),
    )
    cv2.imwrite(
        str(out_dir / "cached_projected_on_viewB.png"),
        _draw_points(img_b_bgr, uv_b_proj[vis_ab], color=(0, 200, 255)),
    )
    cv2.imwrite(
        str(out_dir / "cached_projected_on_viewA.png"),
        _draw_points(img_a_bgr, uv_a_proj[vis_ba], color=(0, 200, 255)),
    )
    cv2.imwrite(
        str(out_dir / "residual_A_to_B.png"),
        _draw_residual_overlay(
            img_b_bgr,
            measured=gt_b[vis_ab],
            projected=uv_b_proj[vis_ab],
            errors=err_ab_v,
            max_draw=args.max_draw,
        ),
    )
    cv2.imwrite(
        str(out_dir / "residual_B_to_A.png"),
        _draw_residual_overlay(
            img_a_bgr,
            measured=gt_a[vis_ba],
            projected=uv_a_proj[vis_ba],
            errors=err_ba_v,
            max_draw=args.max_draw,
        ),
    )

    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("# Endomapper Dense cached-depth projection inspection\n")
        f.write("depth_source=cached_depth_keypoints\n")
        f.write("depth_sampling_runtime=false\n")
        f.write("depth_scale_alignment_runtime=false\n")
        f.write("gt_correspondences_source=colmap_shared_point3D_ids\n")
        f.write(f"scene={args.scene}\n")
        f.write(f"image_a={name_a}\n")
        f.write(f"image_b={name_b}\n")
        f.write(f"num_kpts_A={len(kpts_a)}\n")
        f.write(f"num_kpts_B={len(kpts_b)}\n")
        f.write(f"gt_matches_after_crop={total_gt}\n")
        f.write(f"depth_assign_radius_px={args.depth_assign_radius}\n")
        f.write(f"valid_cached_depth_kpts_A={int(np.sum(v_kp_a))}\n")
        f.write(f"valid_cached_depth_kpts_B={int(np.sum(v_kp_b))}\n")
        f.write(f"gt_depth_assigned_A={int(np.sum(valid_gt_a))}\n")
        f.write(f"gt_depth_assigned_B={int(np.sum(valid_gt_b))}\n")
        f.write(f"gt_to_cached_nn_dist_A={_summary_stats(nn_dist_a)}\n")
        f.write(f"gt_to_cached_nn_dist_B={_summary_stats(nn_dist_b)}\n")
        f.write(f"visible_A_to_B={int(np.sum(vis_ab))}\n")
        f.write(f"visible_B_to_A={int(np.sum(vis_ba))}\n")
        f.write(f"residual_px_A_to_B_visible={_summary_stats(err_ab_v)}\n")
        f.write(f"residual_px_B_to_A_visible={_summary_stats(err_ba_v)}\n")
        f.write(f"recover_pct_A_to_B@1px={recover_ab[1]}\n")
        f.write(f"recover_pct_A_to_B@2px={recover_ab[2]}\n")
        f.write(f"recover_pct_A_to_B@3px={recover_ab[3]}\n")
        f.write(f"recover_pct_A_to_B@4px={recover_ab[4]}\n")
        f.write(f"recover_pct_A_to_B@5px={recover_ab[5]}\n")
        f.write(f"recover_pct_B_to_A@1px={recover_ba[1]}\n")
        f.write(f"recover_pct_B_to_A@2px={recover_ba[2]}\n")
        f.write(f"recover_pct_B_to_A@3px={recover_ba[3]}\n")
        f.write(f"recover_pct_B_to_A@4px={recover_ba[4]}\n")
        f.write(f"recover_pct_B_to_A@5px={recover_ba[5]}\n")

    print(f"[ok] scene={args.scene}, pair=({name_a}, {name_b})")
    print(
        f"[ok] gt_after_crop={total_gt} "
        f"depth_assigned A={int(np.sum(valid_gt_a))} B={int(np.sum(valid_gt_b))}"
    )
    print(
        f"[ok] visible reprojections A->B={int(np.sum(vis_ab))} "
        f"B->A={int(np.sum(vis_ba))}"
    )
    print(
        f"[ok] recover% @1/2/3/4/5px A->B="
        f"{recover_ab[1]:.2f}/{recover_ab[2]:.2f}/{recover_ab[3]:.2f}/{recover_ab[4]:.2f}/{recover_ab[5]:.2f}"
    )
    print(
        f"[ok] recover% @1/2/3/4/5px B->A="
        f"{recover_ba[1]:.2f}/{recover_ba[2]:.2f}/{recover_ba[3]:.2f}/{recover_ba[4]:.2f}/{recover_ba[5]:.2f}"
    )
    print(f"[ok] outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
