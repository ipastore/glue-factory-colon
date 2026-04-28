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


def _get_valid_keypoint_mask_for_view(
    view_idx: int, ref: torch.Tensor, *containers: Dict
) -> torch.Tensor:
    score_key = f"keypoint_scores{view_idx}"
    valid_key = f"valid_3D_mask{view_idx}"
    for container in containers:
        if not container:
            continue
        scores = container.get(score_key)
        if scores is not None:
            scores = scores.to(ref.device)
            if scores.dtype == torch.bool:
                return scores
            # Support both zero-padded and -1-masked conventions.
            return scores > (-1 if torch.any(scores < 0) else 0)
        valid = container.get(valid_key)
        if valid is not None:
            return valid.to(device=ref.device, dtype=torch.bool)
    return torch.ones_like(ref, dtype=torch.bool)


def make_compare_lg_oob_figures(
    pred_epoch: Dict,
    data_: Dict,
    pred_oob: Optional[Dict] = None,
    pred_official: Optional[Dict] = None,
    gt: Optional[Dict] = None,
    n_pairs: int = 2,
    epoch_idx: Optional[int] = None,
    plot_ignored_predictions: bool = False,
):
    """Plot LG_epochX (top) vs LG_OOB (bottom) with matches colored by correctness."""
    pred_epoch = _unwrap_pred(pred_epoch)
    pred_oob = _unwrap_pred(pred_oob)
    pred_official = _unwrap_pred(pred_official)
    if not pred_oob and not pred_official:
        raise ValueError("pred_oob or pred_official is required to draw the LG comparison.")

    pred_epoch = batch_to_device(pred_epoch, "cpu", non_blocking=False)
    pred_oob = batch_to_device(pred_oob, "cpu", non_blocking=False) if pred_oob else {}
    pred_official = (
        batch_to_device(pred_official, "cpu", non_blocking=False)
        if pred_official
        else {}
    )
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if gt is not None:
        gt = batch_to_device(gt, "cpu", non_blocking=False)

    gt_matches0 = gt["matches0"] if gt and "matches0" in gt else pred_epoch["gt_matches0"]
    overlap = data.get("overlap_0to1")
    map_pos0_mask = gt.get("mask_pos_3d_map0") if gt else None
    reproj_pos0_mask = gt.get("mask_pos_reproj0") if gt else None
    pad_mask0 = _get_valid_keypoint_mask_for_view(0, gt_matches0, data, pred_epoch)
    pad_mask1 = _get_valid_keypoint_mask_for_view(
        1,
        pred_epoch.get("matches1", pred_epoch["keypoints1"][..., 0]),
        data,
        pred_epoch,
    )

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])
    kp0_epoch, kp1_epoch = pred_epoch["keypoints0"], pred_epoch["keypoints1"]

    figs = []
    metrics_epoch = matcher_metrics(pred_epoch, {"gt_matches0": gt_matches0})
    metrics_oob = matcher_metrics(pred_oob, {"gt_matches0": gt_matches0}) if pred_oob else None
    metrics_official = (
        matcher_metrics(pred_official, {"gt_matches0": gt_matches0})
        if pred_official
        else None
    )
    for i in range(n_pairs):
        imgs = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        h, w = imgs[0].shape[:2]

        def _draw_compare_figure(baseline_pred, baseline_metrics, baseline_label):
            fig, axes = plot_image_grid(
                [imgs, imgs],
                return_fig=True,
                set_lim=True,
                dpi=300,
                pad=0.05,
            )
            fig.set_size_inches(2 * w / 300, 2 * h / 300 * 1.2)

            for row, (kp0, kp1, matches) in enumerate(
                [
                    (kp0_epoch[i], kp1_epoch[i], pred_epoch["matches0"][i]),
                    (
                        baseline_pred["keypoints0"][i],
                        baseline_pred["keypoints1"][i],
                        baseline_pred["matches0"][i],
                    ),
                ]
            ):
                if plot_ignored_predictions:
                    valid = (matches > -1) & (gt_matches0[i] >= -2)
                else:
                    valid = (matches > -1) & (gt_matches0[i] >= -1)
                valid = valid & pad_mask0[i]
                target_valid = pad_mask1[i].gather(0, matches.clamp(min=0).long())
                valid = valid & target_valid
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
                plot_keypoints(
                    [kp0[pad_mask0[i]], kp1[pad_mask1[i]]],
                    axes=axes[row],
                    colors="royalblue",
                )
                plot_matches(
                    kpm0,
                    kpm1,
                    color=colors,
                    axes=axes[row],
                    a=0.5,
                    lw=1.0,
                    ps=0.0,
                )

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
            axes[1][0].set_title(
                f"{name} | {baseline_label} | {info_suffix}", fontsize=8, loc="left"
            )
            axes[1][1].set_title(
                (
                    f"P:{baseline_metrics['match_precision'][i]:.2f} "
                    f"R:{baseline_metrics['match_recall'][i]:.2f} "
                    f"A:{baseline_metrics['accuracy'][i]:.2f} "
                    f"AP:{baseline_metrics['average_precision'][i]:.2f}"
                ),
                fontsize=8,
                loc="right",
            )
            fig.subplots_adjust(top=0.93, hspace=0.02, left=0.01, right=0.99)
            return fig

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

        fig_entry = {}
        if pred_oob:
            fig_entry["compare_lg_oob"] = _draw_compare_figure(
                pred_oob, metrics_oob, "OOB"
            )
        if pred_official:
            fig_entry["compare_lg_official"] = _draw_compare_figure(
                pred_official, metrics_official, "official"
            )
        figs.append(fig_entry)
    return figs
