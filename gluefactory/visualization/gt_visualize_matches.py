import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from ..utils.tensor import batch_to_device
from ..utils.image import get_pixel_grid, grid_sample, normalize_coords
from .viz2d import plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches
from matplotlib.patches import Patch



def _split_source_masks(map_mask, reproj_mask, pad_mask):
    """Keep masks per source, removing pad and map overlap."""
    map_mask_v = map_mask.to(torch.bool) & pad_mask
    reproj_mask_v = (reproj_mask.to(torch.bool) & pad_mask) & ~map_mask_v
    return map_mask_v, reproj_mask_v


def _compute_view_stats(gt_m, pred_m, pad_mask):
    """Compute per-view stats and masks for FN/TN counting."""
    stats_mask = pad_mask & (gt_m != -2)
    gt_stats = gt_m[stats_mask]
    pred_stats = pred_m[stats_mask]
    fn_mask = (gt_stats > -1) & (pred_stats == -1)
    tn_mask = (gt_stats == -1) & (pred_stats == -1)
    return {
        "stats_mask": stats_mask,
        "fn_mask": fn_mask,
        "tn_mask": tn_mask,
        "num_fn": int(fn_mask.sum()),
        "num_tn": int(tn_mask.sum()),
    }


def _to_hwc_image(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image


def _get_view_title(data, view_key, idx, fallback):
    view = data.get(view_key)
    if view is None:
        return fallback
    image_name = view.get("image_name")
    if image_name is None:
        return fallback
    if isinstance(image_name, (list, tuple)):
        name = image_name[idx]
    else:
        name = image_name[idx] if hasattr(image_name, "__getitem__") else image_name
    if isinstance(name, torch.Tensor):
        name = name.item() if name.ndim == 0 else name.tolist()
    return Path(str(name)).stem


def _get_pad_mask(data, match_key):
    score_key = f"keypoint_scores{match_key[-1]}"
    pad_mask = data.get(score_key)
    if pad_mask is not None:
        return pad_mask > 0
    return None


def _sample_dense_metric_at_keypoints(pred, data, key, idx, view_idx, pad_mask):
    kpts = data[f"keypoints{view_idx}"][idx]
    if pad_mask is None:
        valid = torch.ones(kpts.shape[0], dtype=torch.bool)
    else:
        valid = pad_mask[idx]
    if not valid.any():
        return kpts[valid], torch.empty(0)
    coords = normalize_coords(
        kpts[valid][None], data[f"view{view_idx}"]["image"][idx].shape[-2:]
    )
    values = grid_sample(pred[f"{key}{view_idx}"][idx : idx + 1, None], coords[:, None])[
        0, 0, 0
    ]
    return kpts[valid], values


def _roma_visible_panels(pred, data, idx):
    image0 = data["view0"]["image"][idx : idx + 1]
    image1 = data["view1"]["image"][idx : idx + 1]
    certainty0 = pred["certainty0"][idx : idx + 1, None]
    certainty1 = pred["certainty1"][idx : idx + 1, None]
    q_coords0 = get_pixel_grid(fmap=pred["warp0"][idx : idx + 1], normalized=True)
    q_coords1 = get_pixel_grid(fmap=pred["warp1"][idx : idx + 1], normalized=True)

    image_0to0 = grid_sample(image0, q_coords0)
    image_1to1 = grid_sample(image1, q_coords1)
    image_1to0 = grid_sample(image1, pred["warp0"][idx : idx + 1])
    image_0to1 = grid_sample(image0, pred["warp1"][idx : idx + 1])

    white0 = torch.ones_like(certainty0)
    white1 = torch.ones_like(certainty1)

    return {
        "image0": image0[0],
        "image1": image1[0],
        "image_1to0": image_1to0[0],
        "image_0to1": image_0to1[0],
        "visible_0to0": (certainty0 * image_0to0 + (1 - certainty0) * white0)[0],
        "visible_1to1": (certainty1 * image_1to1 + (1 - certainty1) * white1)[0],
        "visible_1to0": (certainty0 * image_1to0 + (1 - certainty0) * white0)[0],
        "visible_0to1": (certainty1 * image_0to1 + (1 - certainty1) * white1)[0],
    }


def make_gt_roma_raw_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        imgs = [
            _to_hwc_image(data["view0"]["image"][i]),
            _to_hwc_image(data["view1"]["image"][i]),
        ]
        fig, axes = plot_image_grid(
            [imgs],
            titles=[[title0, title1]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        figs.append(fig)
    return figs


def make_gt_roma_demo_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    pure_warp_figs = []
    directional_figs = {"1to0": [], "0to1": []}
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        panels = _roma_visible_panels(pred, data, i)
        fig_1to0, _ = plot_image_grid(
            [[_to_hwc_image(panels["image0"]), _to_hwc_image(panels["visible_1to0"])]],
            titles=[[title0, f"{title1} warp"]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig_0to1, _ = plot_image_grid(
            [[_to_hwc_image(panels["image1"]), _to_hwc_image(panels["visible_0to1"])]],
            titles=[[title1, f"{title0} warp"]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig_pure_warp, _ = plot_image_grid(
            [[
                _to_hwc_image(panels["image_1to0"]),
                _to_hwc_image(panels["image_0to1"]),
            ]],
            titles=[[f"{title1} warp", f"{title0} warp"]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        pure_warp_figs.append(fig_pure_warp)
        directional_figs["1to0"].append(fig_1to0)
        directional_figs["0to1"].append(fig_0to1)
    return pure_warp_figs, directional_figs


def _make_dense_metric_heatmap_fig(pred, data, key, idx, title, vmax=None):
    title0 = _get_view_title(data, "view0", idx, "image0")
    title1 = _get_view_title(data, "view1", idx, "image1")
    metric0 = pred[f"{key}0"][idx].detach()
    metric1 = pred[f"{key}1"][idx].detach()
    fig, axes = plot_image_grid(
        [[metric0, metric1]],
        titles=[[title0, title1]],
        cmaps=["turbo", "turbo"],
        return_fig=True,
        set_lim=True,
        dpi=300,
        pad=0.05,
    )
    axes[0][0].images[0].set_clim(vmin=0.0, vmax=vmax)
    axes[0][1].images[0].set_clim(vmin=0.0, vmax=vmax)
    return fig


def _make_dense_metric_colormap_fig(pred, data, key, idx, title, vmin=0.0, vmax=None):
    image0 = _to_hwc_image(data["view0"]["image"][idx])
    image1 = _to_hwc_image(data["view1"]["image"][idx])
    title0 = _get_view_title(data, "view0", idx, "image0")
    title1 = _get_view_title(data, "view1", idx, "image1")
    metric0 = pred[f"{key}0"][idx].detach()
    metric1 = pred[f"{key}1"][idx].detach()
    fig, axes = plot_image_grid(
        [[image0, image1]],
        titles=[[title0, title1]],
        return_fig=True,
        set_lim=True,
        dpi=300,
        pad=0.05,
    )
    im0 = axes[0][0].imshow(metric0, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.65)
    im1 = axes[0][1].imshow(metric1, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.65)
    cbar1 = fig.colorbar(im1, ax=axes[0][1], fraction=0.025, pad=0.01)
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_label(key, fontsize=7)
    return fig


def _make_dense_metric_custom_colormap_fig(
    pred, data, key, idx, cbar_label, value_fn, norm, ticks=None, ticklabels=None
):
    image0 = _to_hwc_image(data["view0"]["image"][idx])
    image1 = _to_hwc_image(data["view1"]["image"][idx])
    title0 = _get_view_title(data, "view0", idx, "image0")
    title1 = _get_view_title(data, "view1", idx, "image1")
    metric0 = value_fn(pred[f"{key}0"][idx].detach())
    metric1 = value_fn(pred[f"{key}1"][idx].detach())
    fig, axes = plot_image_grid(
        [[image0, image1]],
        titles=[[title0, title1]],
        return_fig=True,
        set_lim=True,
        dpi=300,
        pad=0.05,
    )
    axes[0][0].imshow(metric0, cmap="turbo", norm=norm, alpha=0.65)
    im1 = axes[0][1].imshow(metric1, cmap="turbo", norm=norm, alpha=0.65)
    cbar = fig.colorbar(im1, ax=axes[0][1], fraction=0.025, pad=0.01, ticks=ticks)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(cbar_label, fontsize=7)
    return fig


def make_gt_roma_certainty_heatmap_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_dense_metric_colormap_fig(
                pred,
                data,
                "certainty",
                i,
                "RoMa certainty heatmaps",
                vmin=0.0,
                vmax=1.0,
            )
        )
    return figs


def make_gt_roma_cycle_error_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in pred or "cycle_error1" not in pred:
        return []
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    vmax = 5.0
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_dense_metric_colormap_fig(
                pred,
                data,
                "cycle_error",
                i,
                "RoMa cycle error heatmaps",
                vmin=0.0,
                vmax=vmax,
            )
        )
    return figs


def make_gt_roma_certainty_heatmap_log_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    ticks = [2.0, 3.0, 3.30103, 4.0, 5.0, 6.0]
    ticklabels = ["0.01", "0.001", "0.0005", "0.0001", "0.00001", "0.000001"]
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_dense_metric_custom_colormap_fig(
                pred,
                data,
                "certainty",
                i,
                "-log10(certainty)",
                lambda values: (-torch.log10(values.clamp(1e-6, 1e-2))).detach(),
                mcolors.Normalize(vmin=2.0, vmax=6.0),
                ticks=ticks,
                ticklabels=ticklabels,
            )
        )
    return figs


def make_gt_roma_certainty_heatmap_log_wide_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    ticks = [0.0, 1.0, 2.0, 3.0, 4.0]
    ticklabels = ["1", "0.1", "0.01", "0.001", "0.0001"]
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_dense_metric_custom_colormap_fig(
                pred,
                data,
                "certainty",
                i,
                "-log10(certainty)",
                lambda values: (-torch.log10(values.clamp(1e-4, 1.0))).detach(),
                mcolors.Normalize(vmin=0.0, vmax=4.0),
                ticks=ticks,
                ticklabels=ticklabels,
            )
        )
    return figs


def make_gt_roma_cycle_error_heatmap_log_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in pred or "cycle_error1" not in pred:
        return []
    ticks = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
    ticklabels = ["0.25", "0.5", "1", "2", "5", "10", "20", "30", "50", "75", "100"]
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_dense_metric_custom_colormap_fig(
                pred,
                data,
                "cycle_error",
                i,
                "cycle error (px)",
                lambda values: values.clamp_min(0.25).detach(),
                mcolors.LogNorm(vmin=0.25, vmax=100.0),
                ticks=ticks,
                ticklabels=ticklabels,
            )
        )
    return figs


def _make_sparse_metric_keypoints_fig(pred, data, key, idx, vmin, vmax, cbar_label):
    title0 = _get_view_title(data, "view0", idx, "image0")
    title1 = _get_view_title(data, "view1", idx, "image1")
    fig, axes = plot_image_grid(
        [[
            _to_hwc_image(data["view0"]["image"][idx]),
            _to_hwc_image(data["view1"]["image"][idx]),
        ]],
        titles=[[title0, title1]],
        return_fig=True,
        set_lim=True,
        dpi=300,
        pad=0.05,
    )
    fig.set_size_inches(fig.get_size_inches()[0] * 1.1, fig.get_size_inches()[1])
    fig.subplots_adjust(right=0.86, top=0.95, bottom=0.03)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    pad_mask0 = _get_pad_mask(data, "matches0")
    pad_mask1 = _get_pad_mask(data, "matches1")
    for view_idx, ax, pad_mask in zip((0, 1), axes[0], (pad_mask0, pad_mask1)):
        kpts, values = _sample_dense_metric_at_keypoints(
            pred, data, key, idx, view_idx, pad_mask
        )
        if values.numel():
            ax.scatter(
                kpts[:, 0].detach().numpy(),
                kpts[:, 1].detach().numpy(),
                c=values.detach().numpy(),
                cmap="turbo",
                norm=norm,
                s=4,
                alpha=0.9,
                linewidths=0,
            )
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(cbar_label, fontsize=7)
    return fig


def _make_sparse_metric_keypoints_custom_fig(
    pred, data, key, idx, cbar_label, value_fn, norm, ticks=None, ticklabels=None
):
    title0 = _get_view_title(data, "view0", idx, "image0")
    title1 = _get_view_title(data, "view1", idx, "image1")
    fig, axes = plot_image_grid(
        [[
            _to_hwc_image(data["view0"]["image"][idx]),
            _to_hwc_image(data["view1"]["image"][idx]),
        ]],
        titles=[[title0, title1]],
        return_fig=True,
        set_lim=True,
        dpi=300,
        pad=0.05,
    )
    fig.set_size_inches(fig.get_size_inches()[0] * 1.1, fig.get_size_inches()[1])
    fig.subplots_adjust(right=0.86, top=0.95, bottom=0.03)
    pad_mask0 = _get_pad_mask(data, "matches0")
    pad_mask1 = _get_pad_mask(data, "matches1")
    for view_idx, ax, pad_mask in zip((0, 1), axes[0], (pad_mask0, pad_mask1)):
        kpts, values = _sample_dense_metric_at_keypoints(
            pred, data, key, idx, view_idx, pad_mask
        )
        if values.numel():
            values = value_fn(values)
            ax.scatter(
                kpts[:, 0].detach().numpy(),
                kpts[:, 1].detach().numpy(),
                c=values.detach().numpy(),
                cmap="turbo",
                norm=norm,
                s=4,
                alpha=0.9,
                linewidths=0,
            )
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(cbar_label, fontsize=7)
    return fig


def make_gt_roma_keypoints_certainty_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_sparse_metric_keypoints_fig(
                pred, data, "certainty", i, 0.0, 1.0, "certainty"
            )
        )
    return figs


def make_gt_roma_keypoints_cycle_error_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in pred or "cycle_error1" not in pred:
        return []
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_sparse_metric_keypoints_fig(
                pred, data, "cycle_error", i, 0.0, 5.0, "cycle error (px)"
            )
        )
    return figs


def make_gt_roma_keypoints_certainty_log_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    ticks = [2.0, 3.0, 3.30103, 4.0, 5.0, 6.0]
    ticklabels = ["0.01", "0.001", "0.0005", "0.0001", "0.00001", "0.000001"]
    for i in range(n_pairs):
        figs.append(
            _make_sparse_metric_keypoints_custom_fig(
                pred,
                data,
                "certainty",
                i,
                "-log10(certainty)",
                lambda values: (-torch.log10(values.clamp(1e-6, 1e-2))).detach(),
                mcolors.Normalize(vmin=2.0, vmax=6.0),
                ticks=ticks,
                ticklabels=ticklabels,
            )
        )
    return figs


def make_gt_roma_keypoints_cycle_error_log_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in pred or "cycle_error1" not in pred:
        return []
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    ticks = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
    ticklabels = ["0.25", "0.5", "1", "2", "5", "10", "20", "30", "50", "75", "100"]
    figs = []
    for i in range(n_pairs):
        figs.append(
            _make_sparse_metric_keypoints_custom_fig(
                pred,
                data,
                "cycle_error",
                i,
                "cycle error (px)",
                lambda values: values.clamp_min(0.25).detach(),
                mcolors.LogNorm(vmin=0.25, vmax=100.0),
                ticks=ticks,
                ticklabels=ticklabels,
            )
        )
    return figs


def make_gt_roma_matches_certainty_figs(gt_, data_, n_pairs=2):
    gt = batch_to_device(gt_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0
    else:
        pad_mask0 = torch.ones_like(gt["matches0"], dtype=torch.bool)
    if pad_mask1 is not None:
        pad_mask1 = pad_mask1 > 0
    else:
        pad_mask1 = torch.ones_like(gt["matches1"], dtype=torch.bool)

    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        fig, axes = plot_image_grid(
            [[
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ], [
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ]],
            titles=[[title0, title1], [None, None]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(
            fig.get_size_inches()[0] * 1.18, fig.get_size_inches()[1] * 1.18
        )
        fig.subplots_adjust(right=0.86, hspace=0.0, top=0.97, bottom=0.03)
        match_mask0 = (gt["matches0"][i] > -1) & pad_mask0[i]
        idx0 = torch.nonzero(match_mask0, as_tuple=False).squeeze(-1)
        if idx0.numel():
            idx1 = gt["matches0"][i][idx0].long()
            scores0 = gt["matching_scores0"][i][idx0].numpy()
            colors0 = [
                tuple(c) for c in cm.turbo(np.clip(scores0, 0.0, 1.0)).tolist()
            ]
            plot_matches(
                kp0[i][idx0].numpy(),
                kp1[i][idx1].numpy(),
                color=colors0,
                axes=axes[0],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
        match_mask1 = (gt["matches1"][i] > -1) & pad_mask1[i]
        idx1 = torch.nonzero(match_mask1, as_tuple=False).squeeze(-1)
        if idx1.numel():
            idx0_from_1 = gt["matches1"][i][idx1].long()
            scores1 = gt["matching_scores1"][i][idx1].numpy()
            colors1 = [
                tuple(c) for c in cm.turbo(np.clip(scores1, 0.0, 1.0)).tolist()
            ]
            plot_matches(
                kp0[i][idx0_from_1].numpy(),
                kp1[i][idx1].numpy(),
                color=colors1,
                axes=axes[1],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("certainty", fontsize=7)

        figs.append(fig)
    return figs


def make_gt_roma_matches_pred_certainty_figs(pred_, data_, n_pairs=2):
    return make_gt_roma_matches_certainty_figs(pred_, data_, n_pairs=n_pairs)


def _mutual_intersection_pairs(matches0, matches1, pad_mask0, pad_mask1):
    idx0 = torch.nonzero((matches0 > -1) & pad_mask0, as_tuple=False).squeeze(-1)
    if idx0.numel() == 0:
        return idx0, idx0
    idx1 = matches0[idx0].long()
    valid = pad_mask1[idx1] & (matches1[idx1] == idx0)
    return idx0[valid], idx1[valid]


def make_gt_roma_matches_certainty_intersection_figs(gt_, data_, n_pairs=2):
    gt = batch_to_device(gt_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0
    else:
        pad_mask0 = torch.ones_like(gt["matches0"], dtype=torch.bool)
    if pad_mask1 is not None:
        pad_mask1 = pad_mask1 > 0
    else:
        pad_mask1 = torch.ones_like(gt["matches1"], dtype=torch.bool)

    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        fig, axes = plot_image_grid(
            [[
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ], [
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ]],
            titles=[[title0, title1], [None, None]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(
            fig.get_size_inches()[0] * 1.18, fig.get_size_inches()[1] * 1.18
        )
        fig.subplots_adjust(right=0.86, hspace=0.0, top=0.97, bottom=0.03)
        idx0, idx1 = _mutual_intersection_pairs(
            gt["matches0"][i], gt["matches1"][i], pad_mask0[i], pad_mask1[i]
        )
        if idx0.numel():
            scores0 = gt["matching_scores0"][i][idx0].numpy()
            colors0 = [tuple(c) for c in cm.turbo(np.clip(scores0, 0.0, 1.0)).tolist()]
            plot_matches(
                kp0[i][idx0].numpy(),
                kp1[i][idx1].numpy(),
                color=colors0,
                axes=axes[0],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            scores1 = gt["matching_scores1"][i][idx1].numpy()
            colors1 = [tuple(c) for c in cm.turbo(np.clip(scores1, 0.0, 1.0)).tolist()]
            plot_matches(
                kp0[i][idx0].numpy(),
                kp1[i][idx1].numpy(),
                color=colors1,
                axes=axes[1],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("certainty", fontsize=7)
        figs.append(fig)
    return figs


def make_gt_roma_matches_pred_certainty_intersection_figs(pred_, data_, n_pairs=2):
    return make_gt_roma_matches_certainty_intersection_figs(
        pred_, data_, n_pairs=n_pairs
    )


def make_gt_roma_keypoints_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        fig, axes = plot_image_grid(
            [[
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ]],
            titles=[[title0, title1]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        plot_keypoints(
            [pred["keypoints0"][i], pred["keypoints1"][i]],
            axes=axes[0],
            colors=["lime", "cyan"],
            ps=[1.5, 1.5],
        )
        fig.subplots_adjust(top=0.9)
        figs.append(fig)
    return figs


def make_gt_roma_matches_cycle_error_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in pred:
        return []
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0
    else:
        pad_mask0 = torch.ones_like(pred["matches0"], dtype=torch.bool)
    if pad_mask1 is not None:
        pad_mask1 = pad_mask1 > 0
    else:
        pad_mask1 = torch.ones_like(pred["matches1"], dtype=torch.bool)

    vmax = 5.0
    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        fig, axes = plot_image_grid(
            [[
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ], [
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ]],
            titles=[[title0, title1], [None, None]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(
            fig.get_size_inches()[0] * 1.18, fig.get_size_inches()[1] * 1.18
        )
        fig.subplots_adjust(right=0.86, hspace=0.0, top=0.97, bottom=0.03)
        match_mask0 = (pred["matches0"][i] > -1) & pad_mask0[i]
        idx0 = torch.nonzero(match_mask0, as_tuple=False).squeeze(-1)
        if idx0.numel():
            idx1 = pred["matches0"][i][idx0].long()
            kpts0 = data["keypoints0"][i][idx0]
            coords0 = normalize_coords(
                kpts0[None], data["view0"]["image"][i].shape[-2:]
            )
            cycle_scores0 = grid_sample(
                pred["cycle_error0"][i : i + 1, None], coords0[:, None]
            )[0, 0, 0].detach().numpy()
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            colors0 = [tuple(c) for c in cm.turbo(norm(cycle_scores0)).tolist()]
            plot_matches(
                data["keypoints0"][i][idx0].detach().numpy(),
                data["keypoints1"][i][idx1].detach().numpy(),
                color=colors0,
                axes=axes[0],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
        match_mask1 = (pred["matches1"][i] > -1) & pad_mask1[i]
        idx1 = torch.nonzero(match_mask1, as_tuple=False).squeeze(-1)
        if idx1.numel():
            idx0_from_1 = pred["matches1"][i][idx1].long()
            kpts1 = data["keypoints1"][i][idx1]
            coords1 = normalize_coords(
                kpts1[None], data["view1"]["image"][i].shape[-2:]
            )
            cycle_scores1 = grid_sample(
                pred["cycle_error1"][i : i + 1, None], coords1[:, None]
            )[0, 0, 0].detach().numpy()
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            colors1 = [tuple(c) for c in cm.turbo(norm(cycle_scores1)).tolist()]
            plot_matches(
                data["keypoints0"][i][idx0_from_1].detach().numpy(),
                data["keypoints1"][i][idx1].detach().numpy(),
                color=colors1,
                axes=axes[1],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("cycle error (px)", fontsize=7)
        figs.append(fig)
    return figs


def make_gt_roma_matches_gt_cycle_error_figs(gt_, roma_pred_, data_, n_pairs=2):
    gt = batch_to_device(gt_, "cpu", non_blocking=False)
    roma_pred = batch_to_device(roma_pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in roma_pred:
        return []
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0
    else:
        pad_mask0 = torch.ones_like(gt["matches0"], dtype=torch.bool)
    if pad_mask1 is not None:
        pad_mask1 = pad_mask1 > 0
    else:
        pad_mask1 = torch.ones_like(gt["matches1"], dtype=torch.bool)

    vmax = 5.0
    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        fig, axes = plot_image_grid(
            [[
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ], [
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ]],
            titles=[[title0, title1], [None, None]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(
            fig.get_size_inches()[0] * 1.18, fig.get_size_inches()[1] * 1.18
        )
        fig.subplots_adjust(right=0.86, hspace=0.0, top=0.97, bottom=0.03)
        match_mask0 = (gt["matches0"][i] > -1) & pad_mask0[i]
        idx0 = torch.nonzero(match_mask0, as_tuple=False).squeeze(-1)
        if idx0.numel():
            idx1 = gt["matches0"][i][idx0].long()
            coords0 = normalize_coords(
                data["keypoints0"][i][idx0][None], data["view0"]["image"][i].shape[-2:]
            )
            cycle_scores0 = grid_sample(
                roma_pred["cycle_error0"][i : i + 1, None], coords0[:, None]
            )[0, 0, 0].detach().numpy()
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            colors0 = [tuple(c) for c in cm.turbo(norm(cycle_scores0)).tolist()]
            plot_matches(
                data["keypoints0"][i][idx0].detach().numpy(),
                data["keypoints1"][i][idx1].detach().numpy(),
                color=colors0,
                axes=axes[0],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
        match_mask1 = (gt["matches1"][i] > -1) & pad_mask1[i]
        idx1 = torch.nonzero(match_mask1, as_tuple=False).squeeze(-1)
        if idx1.numel():
            idx0_from_1 = gt["matches1"][i][idx1].long()
            coords1 = normalize_coords(
                data["keypoints1"][i][idx1][None], data["view1"]["image"][i].shape[-2:]
            )
            cycle_scores1 = grid_sample(
                roma_pred["cycle_error1"][i : i + 1, None], coords1[:, None]
            )[0, 0, 0].detach().numpy()
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            colors1 = [tuple(c) for c in cm.turbo(norm(cycle_scores1)).tolist()]
            plot_matches(
                data["keypoints0"][i][idx0_from_1].detach().numpy(),
                data["keypoints1"][i][idx1].detach().numpy(),
                color=colors1,
                axes=axes[1],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("cycle error (px)", fontsize=7)
        figs.append(fig)
    return figs


def make_gt_roma_matches_cycle_error_intersection_figs(pred_, data_, n_pairs=2):
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)
    if "cycle_error0" not in pred or "cycle_error1" not in pred:
        return []
    n_pairs = min(n_pairs, data["view0"]["image"].shape[0])
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0
    else:
        pad_mask0 = torch.ones_like(pred["matches0"], dtype=torch.bool)
    if pad_mask1 is not None:
        pad_mask1 = pad_mask1 > 0
    else:
        pad_mask1 = torch.ones_like(pred["matches1"], dtype=torch.bool)

    vmax = 5.0
    figs = []
    for i in range(n_pairs):
        title0 = _get_view_title(data, "view0", i, "image0")
        title1 = _get_view_title(data, "view1", i, "image1")
        fig, axes = plot_image_grid(
            [[
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ], [
                _to_hwc_image(data["view0"]["image"][i]),
                _to_hwc_image(data["view1"]["image"][i]),
            ]],
            titles=[[title0, title1], [None, None]],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(
            fig.get_size_inches()[0] * 1.18, fig.get_size_inches()[1] * 1.18
        )
        fig.subplots_adjust(right=0.86, hspace=0.0, top=0.97, bottom=0.03)
        idx0, idx1 = _mutual_intersection_pairs(
            pred["matches0"][i], pred["matches1"][i], pad_mask0[i], pad_mask1[i]
        )
        if idx0.numel():
            coords0 = normalize_coords(
                data["keypoints0"][i][idx0][None], data["view0"]["image"][i].shape[-2:]
            )
            cycle_scores0 = grid_sample(
                pred["cycle_error0"][i : i + 1, None], coords0[:, None]
            )[0, 0, 0].detach().numpy()
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            colors0 = [tuple(c) for c in cm.turbo(norm(cycle_scores0)).tolist()]
            plot_matches(
                data["keypoints0"][i][idx0].detach().numpy(),
                data["keypoints1"][i][idx1].detach().numpy(),
                color=colors0,
                axes=axes[0],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            coords1 = normalize_coords(
                data["keypoints1"][i][idx1][None], data["view1"]["image"][i].shape[-2:]
            )
            cycle_scores1 = grid_sample(
                pred["cycle_error1"][i : i + 1, None], coords1[:, None]
            )[0, 0, 0].detach().numpy()
            colors1 = [tuple(c) for c in cm.turbo(norm(cycle_scores1)).tolist()]
            plot_matches(
                data["keypoints0"][i][idx0].detach().numpy(),
                data["keypoints1"][i][idx1].detach().numpy(),
                color=colors1,
                axes=axes[1],
                a=0.7,
                lw=0.8,
                ps=0.5,
            )
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("cycle error (px)", fontsize=7)
        figs.append(fig)
    return figs


def make_gt_pos_neg_ign_figs(gt_, data_, n_pairs=2, pos_th=None, neg_th=None):
    """Return a list of per-pair GT debug figures."""

    gt = batch_to_device(gt_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    gt_m0 = gt["matches0"]
    gt_m1 = gt["matches1"]
    pred_m0 = data.get("matches0", gt_m0)
    pred_m1 = data.get("matches1", gt_m1)
    #TODO: change mask for valid_depth to generalize better or be semantically clear
    pad_mask0 = data["keypoint_scores0"]>0
    pad_mask1 = data["keypoint_scores1"]>0
    overlap = data["overlap_0to1"]
    map_pos0_mask = gt.get("mask_pos_3d_map0")
    map_pos1_mask = gt.get("mask_pos_3d_map1")
    reproj_pos0_mask = gt.get("mask_pos_reproj0")
    reproj_pos1_mask = gt.get("mask_pos_reproj1")

    figs = []
    for i in range(n_pairs):
        
        # Valid are those predictions where both keypoints have a valid keypoint (we don´t look for vallid 3D or valid depth)
        val_pred_mask0 = (pred_m0[i] > -1) & pad_mask0[i]
        m0i_clamepd = pred_m0[i].clamp(min=0)
        val_pred_mask1 = pad_mask1[i].gather(0, m0i_clamepd)
        val_pred_mask = val_pred_mask0 & val_pred_mask1 

        # Filter the matched keypoints
        kpm0, kpm1 = kp0[i][val_pred_mask].numpy(), kp1[i][pred_m0[i][val_pred_mask]].numpy()

        imgs = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        h, w = imgs[0].shape[:2]
        figsize = (2 * w / 300, (h / 300) * 1.2)  # slight headroom for title
        fig, axes = plot_image_grid(
            [imgs],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(figsize[0], figsize[1])

        true_positives = gt_m0[i][val_pred_mask] == pred_m0[i][val_pred_mask]
        # view0_stats = _compute_view_stats(gt_m0[i], pred_m0[i], pad_mask0[i])
        # view1_stats = _compute_view_stats(gt_m1[i], pred_m1[i], pad_mask1[i])
        # num_false_negatives0 = view0_stats["num_fn"]
        # num_true_negatives0 = view0_stats["num_tn"]
        # num_false_negatives1 = view1_stats["num_fn"]
        # num_true_negatives1 = view1_stats["num_tn"]
        # num_true_positives = int(true_positives.sum())
        num_false_positives = int((~true_positives).sum())

        # kpts0_stats = kp0[i][view0_stats["stats_mask"]]
        # kpts1_stats = kp1[i][view1_stats["stats_mask"]]
        # fn_stats0 = view0_stats["fn_mask"]
        # fn_stats1 = view1_stats["fn_mask"]

        kp0_pos_mask = pad_mask0[i] & (gt_m0[i] > -1)
        kp1_pos_mask = pad_mask1[i] & (gt_m1[i] > -1)
        kp0_neg_mask = pad_mask0[i] & (gt_m0[i] == -1)
        kp1_neg_mask = pad_mask1[i] & (gt_m1[i] == -1)
        kp0_ign_mask = pad_mask0[i] & (gt_m0[i] == -2)
        kp1_ign_mask = pad_mask1[i] & (gt_m1[i] == -2)

        kp0_map_mask, kp0_reproj_mask = _split_source_masks(
            map_pos0_mask[i], reproj_pos0_mask[i], pad_mask0[i]
        )
        kp1_map_mask, kp1_reproj_mask = _split_source_masks(
            map_pos1_mask[i], reproj_pos1_mask[i], pad_mask1[i]
        )

        map_pos = int(kp0_map_mask.sum())
        reproj_pos = int(kp0_reproj_mask.sum())

        match_colors = ["limegreen" if tp else "red" for tp in true_positives.tolist()]
        pred_indices0 = torch.nonzero(val_pred_mask, as_tuple=False).squeeze(-1)
        pred_indices1 = pred_m0[i][pred_indices0]
        gt_labels0 = gt_m0[i][pred_indices0]
        gt_labels1 = gt_m1[i][pred_indices1]
        ignored_pair = (gt_labels0 == -2) & (gt_labels1 == -2)
        fp_ignored = int((~true_positives & ignored_pair).sum())
        fp_regular = num_false_positives - fp_ignored
        line_colors = [
            "limegreen" if tp else ("white" if ign else "red")
            for tp, ign in zip(true_positives.tolist(), ignored_pair.tolist())
        ]

        line_colors_rgba = (
            [mcolors.to_rgba(c) for c in line_colors] if len(line_colors) else line_colors
        )

        # # TP and FP, green and red lines
        # plot_matches(
        #     kpm0,
        #     kpm1,
        #     color=line_colors_rgba,
        #     axes=axes[0],
        #     a=0.6,
        #     lw=0.5,
        #     ps=0.0,
        # )
        # # TP and FP, green and red edges
        # plot_keypoints(
        #     [kpm0, kpm1],
        #     axes=axes[0],
        #     edgecolors=[match_colors, match_colors],
        #     facecolors=["none","none"],
        #     ps=[2, 2],
        #     lw=[0.3, 0.3],
        # )

        # TP faces (map vs reprojection)
        plot_keypoints(
            [kp0[i][kp0_reproj_mask], kp1[i][kp1_reproj_mask]],
            axes=axes[0],
            facecolors=["purple", "purple"],
            edgecolors=["none", "none"],
            lw=[1, 1],
            ps=[1, 1],
        )

        # IGNORED face wout edge
        plot_keypoints(
            [kp0[i][kp0_ign_mask], kp1[i][kp1_ign_mask]],
            axes=axes[0],
            facecolors=["black", "black"],
            ps=[1, 1],
            # a=[0.6, 0.6],
        )


        plot_keypoints(
            [kp0[i][kp0_map_mask], kp1[i][kp1_map_mask]],
            axes=axes[0],
            facecolors=["limegreen", "limegreen"],
            ps=[1, 1],
        )

        # # TN face wout edge
        plot_keypoints(
            [kp0[i][kp0_neg_mask], kp1[i][kp1_neg_mask]],
            axes=axes[0],
            facecolors=["blue", "blue"],
            ps=[1, 1],
        )


        # # FN edge
        # plot_keypoints(
        #     [kpts0_stats[fn_stats0], kpts1_stats[fn_stats1]],
        #     axes=axes[0],
        #     edgecolors=["black", "black"],
        #     facecolors=["none", "none"],
        #     ps=[3, 3],
        #     lw=[0.3, 0.3],
        # )

    
        title_line0_parts = [
                f"KP: tot/pos/neg/ign"  #using tabs
                f"ov: {overlap[i]:.2f}",    # Take into account that some keypoints and 3D points could have been truncated: ov =! GT_POS/min(KP3D)
                f"GT_POS map+reproj: {map_pos}+{reproj_pos}",
                f"n_pred: {val_pred_mask.sum()}"
            ]
        if pos_th is not None:
            title_line0_parts.append(f"pos_th: {pos_th:g}")
        if neg_th is not None:
            title_line0_parts.append(f"neg_th: {neg_th:g}")
        title_line0 = " | ".join(title_line0_parts)
        title_line1_parts = [
            f"IMG0-> "
            f"KP: {pad_mask0[i].sum()}/{kp0_pos_mask.sum()}/{kp0_neg_mask.sum()}/{kp0_ign_mask.sum()}",
        ]
        valid_3d_mask0 = data.get("valid_3D_mask0")
        if valid_3d_mask0 is not None:
            title_line1_parts.append(f"KP_3D: {valid_3d_mask0[i].sum()}")
        title_line1 = " | ".join(title_line1_parts)
        title_line2_parts = [
            f"IMG1-> "
            f"KP: {pad_mask1[i].sum()}/{kp1_pos_mask.sum()}/{kp1_neg_mask.sum()}/{kp1_ign_mask.sum()}",
        ]
        valid_3d_mask1 = data.get("valid_3D_mask1")
        if valid_3d_mask1 is not None:
            title_line2_parts.append(f"KP_3D: {valid_3d_mask1[i].sum()}")
        title_line2 = " | ".join(title_line2_parts)
        fig.suptitle(
            f"{title_line0}\n{title_line1}\n{title_line2}", fontsize=8, y=0.99, va="bottom"
        )
        fig.subplots_adjust(top=1.1)
            
        # Add colored legend in a column on the right side
        legend_elements = [
            Patch(facecolor='limegreen', edgecolor='none', label='GT pos map'),
            Patch(facecolor='purple', edgecolor='none', label='GT pos reproj'),
            Patch(facecolor='blue', edgecolor='none', label='GT neg'),
            Patch(facecolor='black', edgecolor='none', label='GT ign'),
            # Patch(facecolor='none', edgecolor='limegreen', label='TP'),
            # Patch(facecolor='none', edgecolor='red', label='FP'),
            # Patch(facecolor='none', edgecolor='lightgray', label='FP ignored'),
            # Patch(facecolor='none', edgecolor='black', label='FN'),
            # Patch(facecolor='none', edgecolor='blue', label='TN'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='center left',
            fontsize=5,
            framealpha=0.7,
            bbox_to_anchor=(1.02, 0.7),
            ncol=1,
        )

        figs.append(fig)

    return figs


def make_gt_pos_neg_ign_roma_figs(gt_, data_, n_pairs=2, pos_th=None, neg_th=None):
    """Return per-pair RoMa GT figures using explicit GT category masks."""

    gt = batch_to_device(gt_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    pad_mask0 = data["keypoint_scores0"] > 0
    pad_mask1 = data["keypoint_scores1"] > 0
    overlap = data.get("overlap_0to1")

    figs = []
    for i in range(n_pairs):
        imgs = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        h, w = imgs[0].shape[:2]
        figsize = (2 * w / 300, (h / 300) * 1.2)
        fig, axes = plot_image_grid(
            [imgs],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(figsize[0], figsize[1])

        masks = {
            "pos": (gt["mask_pos0"][i] & pad_mask0[i], gt["mask_pos1"][i] & pad_mask1[i]),
            "neg_far": (
                gt["mask_neg_far0"][i] & pad_mask0[i],
                gt["mask_neg_far1"][i] & pad_mask1[i],
            ),
            "neg_unreliable": (
                gt["mask_neg_unreliable0"][i] & pad_mask0[i],
                gt["mask_neg_unreliable1"][i] & pad_mask1[i],
            ),
            "ign": (gt["mask_ign0"][i] & pad_mask0[i], gt["mask_ign1"][i] & pad_mask1[i]),
        }
        colors = {
            "pos": "limegreen",
            "neg_far": "#ffd400",
            "neg_unreliable": "magenta",
            "ign": "black",
        }

        for key in ["pos", "neg_far", "neg_unreliable", "ign"]:
            plot_keypoints(
                [kp0[i][masks[key][0]], kp1[i][masks[key][1]]],
                axes=axes[0],
                facecolors=[colors[key], colors[key]],
                edgecolors=["none", "none"],
                ps=[1, 1],
            )

        ov = overlap[i].item() if overlap is not None else float("nan")
        title_parts = [
            f"ov: {ov:.2f}",
            f"pos_th: {pos_th:g}" if pos_th is not None else None,
            f"neg_th: {neg_th:g}" if neg_th is not None else None,
        ]
        title_parts = [p for p in title_parts if p is not None]
        line0 = " | ".join(title_parts)
        line1 = (
            f"IMG0-> KP: {int(pad_mask0[i].sum())}/"
            f"{int(masks['pos'][0].sum())}/{int(masks['neg_far'][0].sum())}/"
            f"{int(masks['neg_unreliable'][0].sum())}/{int(masks['ign'][0].sum())}"
        )
        line2 = (
            f"IMG1-> KP: {int(pad_mask1[i].sum())}/"
            f"{int(masks['pos'][1].sum())}/{int(masks['neg_far'][1].sum())}/"
            f"{int(masks['neg_unreliable'][1].sum())}/{int(masks['ign'][1].sum())}"
        )
        fig.suptitle(
            f"{line0}\n{line1}\n{line2}", fontsize=8, y=0.99, va="bottom"
        )
        fig.subplots_adjust(top=1.1)

        legend_elements = [
            Patch(facecolor="limegreen", edgecolor="none", label="GT pos"),
            Patch(facecolor="#ffd400", edgecolor="none", label="GT neg far"),
            Patch(facecolor="magenta", edgecolor="none", label="GT neg unreliable"),
            Patch(facecolor="black", edgecolor="none", label="GT ign"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="center left",
            fontsize=5,
            framealpha=0.7,
            bbox_to_anchor=(1.02, 0.7),
            ncol=1,
        )
        figs.append(fig)

    return figs

def make_gt_pos_figs(pred_, data_, n_pairs=2, pos_th=None):
    """Return a list of per-pair GT positive figures."""
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    gt_m0 = pred.get("plot_matches0", pred["matches0"])
    overlap = data.get("overlap_0to1")
    val3D_mask0 = data.get("valid_3D_mask0")
    val3D_mask1 = data.get("valid_3D_mask1")
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    if "mask_pos_3d_map0" in pred or "plot_mask_pos_3d_map0" in pred:
        map_pos0_mask = pred.get("plot_mask_pos_3d_map0", pred["mask_pos_3d_map0"])
        reproj_pos0_mask = pred.get("plot_mask_pos_reproj0", pred["mask_pos_reproj0"])
        has_reproj_label = True
    else:
        map_pos0_mask = pred.get("plot_mask_pos0", pred["matches0"] > -1)
        reproj_pos0_mask = torch.zeros_like(map_pos0_mask)
        has_reproj_label = False

    if pad_mask0 is not None:
        pad_mask0 = pad_mask0 > 0
    else:
        pad_mask0 = torch.ones_like(gt_m0, dtype=torch.bool)
    if pad_mask1 is not None:
        pad_mask1 = pad_mask1 > 0
    else:
        pad_mask1 = torch.ones_like(pred["matches1"], dtype=torch.bool)

    figs = []
    for i in range(n_pairs):
        kp0_map_mask, kp0_reproj_mask = _split_source_masks(
            map_pos0_mask[i], reproj_pos0_mask[i], pad_mask0[i]
        )
        map_kp0 = kp0[i][kp0_map_mask].numpy()
        map_idx1 = gt_m0[i][kp0_map_mask].long()
        map_kp1 = kp1[i][map_idx1].numpy()

        reproj_kp0 = kp0[i][kp0_reproj_mask].numpy()
        reproj_idx1 = gt_m0[i][kp0_reproj_mask].long()
        reproj_kp1 = kp1[i][reproj_idx1].numpy()

        imgs = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        h, w = imgs[0].shape[:2]
        figsize = (2 * w / 300, (h / 300) * 1.2)
        fig, axes = plot_image_grid(
            [imgs],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(figsize[0], figsize[1])

        if map_kp0.shape[0]:
            plot_matches(
                map_kp0,
                map_kp1,
                color="limegreen",
                axes=axes[0],
                a=0.5,
                lw=0.5,
                ps=0.5,
            )
        if reproj_kp0.shape[0]:
            plot_matches(
                reproj_kp0,
                reproj_kp1,
                color="purple",
                axes=axes[0],
                a=0.5,
                lw=0.5,
                ps=0.5,
            )

        ov = overlap[i].item() if overlap is not None else float("nan")
        kp0_tot = int(pad_mask0[i].sum())
        kp1_tot = int(pad_mask1[i].sum())
        kp0_3d = int(val3D_mask0[i].sum()) if val3D_mask0 is not None else 0
        kp1_3d = int(val3D_mask1[i].sum()) if val3D_mask1 is not None else 0
        map_pos = int(kp0_map_mask.sum())
        reproj_pos = int(kp0_reproj_mask.sum())
        title_parts = [
            f"ov: {ov:.2f}",
            f"KP0 tot: {kp0_tot} KP0-3D: {kp0_3d}",
            f"KP1 tot: {kp1_tot} KP1-3D: {kp1_3d}",
            (
                f"GT_POS map+reproj: {map_pos}+{reproj_pos}"
                if has_reproj_label
                else f"GT_POS: {map_pos}"
            ),
        ]
        if pos_th is not None:
            title_parts.append(f"pos_th: {pos_th:g}")
        fig.suptitle(" | ".join(title_parts), fontsize=8, y=0.99, va="bottom")
        fig.subplots_adjust(top=1.02)
        legend_elements = [Patch(facecolor='limegreen', edgecolor='none', label='GT pos')]
        if has_reproj_label:
            legend_elements.append(
                Patch(facecolor='purple', edgecolor='none', label='GT pos reproj')
            )
        fig.legend(
            handles=legend_elements,
            loc='center left',
            fontsize=6,
            framealpha=0.7,
            bbox_to_anchor=(1.02, 0.8),
            ncol=1,
        )
        figs.append(fig)

    return figs


def make_gt_pos_sparse_map_figs(pred_, data_, n_pairs=2, pos_th=None):
    del pred_, pos_th
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    if "colmap_xys" not in view0 or "colmap_xys" not in view1:
        return []

    n_pairs = min(n_pairs, view0["image"].shape[0])
    figs = []
    overlap = data.get("overlap_0to1")

    for i in range(n_pairs):
        colmap_xys0 = view0["colmap_xys"][i]
        colmap_xys1 = view1["colmap_xys"][i]
        point3d_ids0 = view0["point3D_ids"][i]
        point3d_ids1 = view1["point3D_ids"][i]
        valid0 = view0.get("valid_3D_mask")
        valid1 = view1.get("valid_3D_mask")
        if valid0 is None:
            valid0 = point3d_ids0 != -1
        else:
            valid0 = valid0[i] & (point3d_ids0 != -1)
        if valid1 is None:
            valid1 = point3d_ids1 != -1
        else:
            valid1 = valid1[i] & (point3d_ids1 != -1)

        ids0 = point3d_ids0[valid0]
        ids1 = point3d_ids1[valid1]
        xys0 = colmap_xys0[valid0]
        xys1 = colmap_xys1[valid1]
        shared_ids, idx0, idx1 = np.intersect1d(
            ids0.numpy(), ids1.numpy(), return_indices=True
        )

        imgs = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        h, w = imgs[0].shape[:2]
        figsize = (2 * w / 300, (h / 300) * 1.2)
        fig, axes = plot_image_grid(
            [imgs],
            return_fig=True,
            set_lim=True,
            dpi=300,
            pad=0.05,
        )
        fig.set_size_inches(figsize[0], figsize[1])

        if len(shared_ids):
            plot_matches(
                xys0[idx0].numpy(),
                xys1[idx1].numpy(),
                color="limegreen",
                axes=axes[0],
                a=0.5,
                lw=0.5,
                ps=0.5,
            )

        ov = overlap[i].item() if overlap is not None else float("nan")
        title_parts = [
            f"ov: {ov:.2f}",
            f"COLMAP0 valid: {int(valid0.sum())}",
            f"COLMAP1 valid: {int(valid1.sum())}",
            f"GT_POS sparse_map: {len(shared_ids)}",
        ]
        fig.suptitle(" | ".join(title_parts), fontsize=8, y=0.99, va="bottom")
        fig.subplots_adjust(top=1.02)
        fig.legend(
            handles=[Patch(facecolor="limegreen", edgecolor="none", label="GT pos sparse map")],
            loc="center left",
            fontsize=6,
            framealpha=0.7,
            bbox_to_anchor=(1.02, 0.8),
            ncol=1,
        )
        figs.append(fig)

    return figs
