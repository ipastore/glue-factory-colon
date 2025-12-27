from __future__ import annotations

from typing import Dict, Optional

import torch

from ..models.utils.metrics import matcher_metrics
from ..utils.tensor import batch_to_device
import matplotlib.colors as mcolors

from .viz2d import cm_RdGn, plot_image_grid, plot_keypoints, plot_matches


def _unwrap_pred(pred: Dict) -> Dict:
    if pred is None:
        return {}
    return pred["0to1"] if "0to1" in pred else pred


def _clean_name(name: str) -> str:
    s = str(name)
    s = s.replace("Seq_", "")
    s = s.replace("Keyframe_", "")
    for ext in [".png", ".pg"]:
        s = s.replace(ext, "")
    s = s.replace("/", "_")
    return s


def make_compare_lg_oob_figures(
    pred_epoch: Dict,
    data_: Dict,
    pred_oob: Optional[Dict] = None,
    gt: Optional[Dict] = None,
    n_pairs: int = 2,
    epoch_idx: Optional[int] = None,
    plot_ignored_predictions: bool = False,
):
    """Plot LG_epochX (top) vs LG_OOB (bottom) with matches colored by correctness."""
    pred_epoch = _unwrap_pred(pred_epoch)
    pred_oob = _unwrap_pred(pred_oob)
    if not pred_oob:
        raise ValueError("pred_oob is required to draw the OOB comparison.")

    pred_epoch = batch_to_device(pred_epoch, "cpu", non_blocking=False)
    pred_oob = batch_to_device(pred_oob, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if gt is not None:
        gt = batch_to_device(gt, "cpu", non_blocking=False)

    gt_matches0 = gt["matches0"] if gt and "matches0" in gt else pred_epoch["gt_matches0"]
    overlap = data.get("overlap_0to1")
    map_pos0_mask = gt.get("mask_pos_3d_map0") if gt else None
    reproj_pos0_mask = gt.get("mask_pos_reproj0") if gt else None
    pad_mask0 = data.get("keypoint_scores0")
    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])
    kp0_epoch, kp1_epoch = pred_epoch["keypoints0"], pred_epoch["keypoints1"]
    kp0_oob, kp1_oob = pred_oob["keypoints0"], pred_oob["keypoints1"]

    figs = []
    metrics_epoch = matcher_metrics(pred_epoch, {"gt_matches0": gt_matches0})
    metrics_oob = matcher_metrics(pred_oob, {"gt_matches0": gt_matches0})
    for i in range(n_pairs):
        imgs = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        fig, axes = plot_image_grid(
            [imgs, imgs],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        h, w = imgs[0].shape[:2]
        fig.set_size_inches(2 * w / 300, 2 * h / 300 * 1.2)

        for row, (kp0, kp1, matches) in enumerate(
            [
                (kp0_epoch[i], kp1_epoch[i], pred_epoch["matches0"][i]),
                (kp0_oob[i], kp1_oob[i], pred_oob["matches0"][i]),
            ]
        ):
            if plot_ignored_predictions:
                valid = (matches > -1) & (gt_matches0[i] >= -2)
            else:
                valid = (matches > -1) & (gt_matches0[i] >= -1)
            kpm0 = kp0[valid]
            kpm1 = kp1[matches[valid]]
            correct = gt_matches0[i][valid] == matches[valid]
            colors = cm_RdGn(correct.float()).tolist()
            if plot_ignored_predictions:
                ignored = gt_matches0[i][valid] == -2
                ignored_idx = torch.nonzero(ignored, as_tuple=False).squeeze(-1)
                if ignored_idx.numel() > 0:
                    ignored_rgba = mcolors.to_rgba("lightgray")
                    for idx in (
                        ignored_idx.tolist()
                        if ignored_idx.ndim > 0
                        else [ignored_idx.item()]
                    ):
                        colors[idx] = ignored_rgba
            plot_keypoints([kp0, kp1], axes=axes[row], colors="royalblue")
            plot_matches(
                kpm0,
                kpm1,
                color=colors,
                axes=axes[row],
                a=0.5,
                lw=1.0,
                ps=0.0,
            )

        names = data.get("names")
        if isinstance(names, (list, tuple)):
            name = names[i] if i < len(names) else names[0]
        elif torch.is_tensor(names):
            name = names[i].item() if names.ndim > 0 else names.item()
        else:
            name = names if names is not None else f"pair_{i}"
        name = _clean_name(name)

        ov = overlap[i].item() if overlap is not None else float("nan")
        info_parts = [f"ov: {ov:.2f}"]
        if map_pos0_mask is not None and reproj_pos0_mask is not None:
            map_mask = map_pos0_mask[i].to(torch.bool)
            reproj_mask = reproj_pos0_mask[i].to(torch.bool)
            if pad_mask0 is not None:
                map_mask = map_mask & pad_mask0[i]
                reproj_mask = (reproj_mask & pad_mask0[i]) & ~map_mask
            info_parts.append(f"GT: {int(map_mask.sum())}+{int(reproj_mask.sum())}")
        info_suffix = " | ".join(info_parts)

        axes[0][0].set_title(
            f"{name} | epoch{epoch_idx} | {info_suffix}", fontsize=8, loc="left"
        )
        axes[0][1].set_title(
            (
                f"P:{metrics_epoch['match_precision'][i]:.2f} "
                f"R:{metrics_epoch['match_recall'][i]:.2f} "
                f"A:{metrics_epoch['accuracy'][i]:.2f} "
                f"AP:{metrics_epoch['average_precision'][i]:.2f}"
            ),
            fontsize=8,
            loc="right",
        )
        axes[1][0].set_title(f"{name} | OOB | {info_suffix}", fontsize=8, loc="left")
        axes[1][1].set_title(
            (
                f"P:{metrics_oob['match_precision'][i]:.2f} "
                f"R:{metrics_oob['match_recall'][i]:.2f} "
                f"A:{metrics_oob['accuracy'][i]:.2f} "
                f"AP:{metrics_oob['average_precision'][i]:.2f}"
            ),
            fontsize=8,
            loc="right",
        )
        fig.subplots_adjust(top=0.93, hspace=0.02, left=0.01, right=0.99)

        figs.append({"compare_lg_oob": fig})
    return figs
