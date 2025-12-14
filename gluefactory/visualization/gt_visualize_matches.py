import torch
import matplotlib.colors as mcolors

from ..utils.tensor import batch_to_device
from .viz2d import plot_image_grid, plot_keypoints, plot_matches
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


def make_gt_debug_figures(gt_, data_, n_pairs=2):
    """Return a list of per-pair GT debug figures."""

    gt = batch_to_device(gt_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    pred_m0 = data["matches0"]
    pred_m1 = data["matches1"]
    gt_m0 = gt["matches0"]
    gt_m1 = gt["matches1"]
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
        view0_stats = _compute_view_stats(gt_m0[i], pred_m0[i], pad_mask0[i])
        view1_stats = _compute_view_stats(gt_m1[i], pred_m1[i], pad_mask1[i])
        num_false_negatives0 = view0_stats["num_fn"]
        num_true_negatives0 = view0_stats["num_tn"]
        num_false_negatives1 = view1_stats["num_fn"]
        num_true_negatives1 = view1_stats["num_tn"]
        num_true_positives = int(true_positives.sum())
        num_false_positives = int((~true_positives).sum())

        kpts0_stats = kp0[i][view0_stats["stats_mask"]]
        kpts1_stats = kp1[i][view1_stats["stats_mask"]]
        fn_stats0 = view0_stats["fn_mask"]
        fn_stats1 = view1_stats["fn_mask"]

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

        # TP and FP, green and red lines
        plot_matches(
            kpm0,
            kpm1,
            color=line_colors_rgba,
            axes=axes[0],
            a=0.6,
            lw=0.5,
            ps=0.0,
        )
        # TP and FP, green and red edges
        plot_keypoints(
            [kpm0, kpm1],
            axes=axes[0],
            edgecolors=[match_colors, match_colors],
            facecolors=["none","none"],
            ps=[2, 2],
            lw=[0.3, 0.3],
        )

        # TP faces (map vs reprojection)
        plot_keypoints(
            [kp0[i][kp0_reproj_mask], kp1[i][kp1_reproj_mask]],
            axes=axes[0],
            facecolors=["purple", "purple"],
            edgecolors=["limegreen", "limegreen"],
            lw=[0.5, 0.5],
            ps=[2, 2],
        )

        plot_keypoints(
            [kp0[i][kp0_map_mask], kp1[i][kp1_map_mask]],
            axes=axes[0],
            facecolors=["limegreen", "limegreen"],
            ps=[2, 2],
        )

        # # TN face wout edge
        plot_keypoints(
            [kp0[i][kp0_neg_mask], kp1[i][kp1_neg_mask]],
            axes=axes[0],
            facecolors=["blue", "blue"],
            ps=[1, 1],
        )

        # IGNORED face wout edge
        plot_keypoints(
            [kp0[i][kp0_ign_mask], kp1[i][kp1_ign_mask]],
            axes=axes[0],
            facecolors=["dimgray", "dimgray"],
            ps=[1, 1],
            # a=[0.6, 0.6],
        )

        # FN edge
        plot_keypoints(
            [kpts0_stats[fn_stats0], kpts1_stats[fn_stats1]],
            axes=axes[0],
            edgecolors=["black", "black"],
            facecolors=["none", "none"],
            ps=[3, 3],
            lw=[0.3, 0.3],
        )

    
        title_line0 = " | ".join(
            [   
                f"KP: tot/pos/neg/ign                                      "  #using tabs
                f"ov: {overlap[i]:.2f}",    # Take into account that some keypoints and 3D points could have been truncated: ov =! GT_POS/min(KP3D)
                f"GT_POS map+reproj: {map_pos}+{reproj_pos}",
                f"n_pred: {val_pred_mask.sum()}"

            ]
        )
        title_line1 = " | ".join(
            [
                f"IMG0-> "    
                f"KP: {pad_mask0[i].sum()}/{kp0_pos_mask.sum()}/{kp0_neg_mask.sum()}/{kp0_ign_mask.sum()}",
                f"KP_3D: {data['valid_3D_mask0'][i].sum()}", 
                f"TP: {num_true_positives}",
                f"FP: {fp_regular}",
                f"FP ign: {fp_ignored}",
                f"FN0: {num_false_negatives0}",
                f"TN0: {num_true_negatives0}",
            ]
        )
        title_line2 = " | ".join(
            [
                f"IMG1-> " 
                f"KP: {pad_mask1[i].sum()}/{kp1_pos_mask.sum()}/{kp1_neg_mask.sum()}/{kp1_ign_mask.sum()}",
                f"KP_3D: {data['valid_3D_mask1'][i].sum()}",
                f"TP: {num_true_positives}",
                f"FP: {fp_regular}",
                f"FP ign: {fp_ignored}",
                f"FN1: {num_false_negatives1}",
                f"TN1: {num_true_negatives1}",
            ]
        )
        fig.suptitle(
            f"{title_line0}\n{title_line1}\n{title_line2}", fontsize=8, y=0.99, va="bottom"
        )
        fig.subplots_adjust(top=1.1)
            
        # Add colored legend in a column on the right side
        legend_elements = [
            Patch(facecolor='limegreen', edgecolor='none', label='GT pos map'),
            Patch(facecolor='purple', edgecolor='limegreen', label='GT pos reproj'),
            Patch(facecolor='blue', edgecolor='none', label='GT neg'),
            Patch(facecolor='dimgray', edgecolor='none', label='GT ign'),
            Patch(facecolor='none', edgecolor='limegreen', label='TP'),
            Patch(facecolor='none', edgecolor='red', label='FP'),
            Patch(facecolor='none', edgecolor='lightgray', label='FP ignored'),
            Patch(facecolor='none', edgecolor='black', label='FN'),
            Patch(facecolor='none', edgecolor='blue', label='TN'),
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

def make_gt_pos_figures(pred_, data_, n_pairs=2):
    """Return a list of per-pair GT positive figures."""
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    gt_m0 = pred["matches0"]
    overlap = data.get("overlap_0to1")
    val3D_mask0 = data.get("valid_3D_mask0")
    val3D_mask1 = data.get("valid_3D_mask1")
    pad_mask0 = data.get("keypoint_scores0")
    pad_mask1 = data.get("keypoint_scores1")
    map_pos0_mask = pred["mask_pos_3d_map0"]
    reproj_pos0_mask = pred["mask_pos_reproj0"]

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
                a=0.7,
                lw=1.0,
                ps=2.5,
            )
        if reproj_kp0.shape[0]:
            plot_matches(
                reproj_kp0,
                reproj_kp1,
                color="purple",
                axes=axes[0],
                a=0.7,
                lw=0.8,
                ps=2.0,
            )

        ov = overlap[i].item() if overlap is not None else float("nan")
        kp0_tot = int(pad_mask0[i].sum())
        kp1_tot = int(pad_mask1[i].sum())
        kp0_3d = int(val3D_mask0[i].sum()) if val3D_mask0 is not None else 0
        kp1_3d = int(val3D_mask1[i].sum()) if val3D_mask1 is not None else 0
        map_pos = int(kp0_map_mask.sum())
        reproj_pos = int(kp0_reproj_mask.sum())
        fig.suptitle(
            " | ".join(
                [
                    f"ov: {ov:.2f}",
                    f"KP0 tot: {kp0_tot} KP0-3D: {kp0_3d}",
                    f"KP1 tot: {kp1_tot} KP1-3D: {kp1_3d}",
                    f"GT_POS map+reproj: {map_pos}+{reproj_pos}",
                ]
            ),
            fontsize=8,
            y=0.99,
            va="bottom",
        )
        fig.subplots_adjust(top=1.02)
        legend_elements = [
            Patch(facecolor='limegreen', edgecolor='none', label='GT pos map'),
            Patch(facecolor='purple', edgecolor='none', label='GT pos reproj'),
        ]
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
