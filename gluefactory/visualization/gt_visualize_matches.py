import torch

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches


def make_gt_debug_figures(pred_, data_, n_pairs=2):
    """Return a list of per-pair GT debug figures."""
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = data["keypoints0"], data["keypoints1"]
    pred_m0 = data["matches0"]
    pred_m1 = data["matches1"]
    val3D_mask0 = data["valid_3D_mask0"]
    val3D_mask1 = data["valid_3D_mask1"]
    gt_m0 = pred["matches0"]
    gt_m1 = pred["matches1"]
    pad_mask0 = data["keypoint_scores0"]>0
    pad_mask1 = data["keypoint_scores1"]>0
    overlap = data["overlap_0to1"]

    figs = []
    for i in range(n_pairs):
        
        # Valid are those predictions where both keypoints have a valid 3D
        val3D_pred_mask0 = (pred_m0[i] > -1) & val3D_mask0[i]
        m0i_clamepd = pred_m0[i].clamp(min=0)
        val3D_pred_mask1 = val3D_mask1[i].gather(0, m0i_clamepd)
        val3D_pred_mask = val3D_pred_mask0 & val3D_pred_mask1 

        # Filter the matched keypoints
        kpm0, kpm1 = kp0[i][val3D_pred_mask].numpy(), kp1[i][pred_m0[i][val3D_pred_mask]].numpy()

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

        true_positives = gt_m0[i][val3D_pred_mask] == pred_m0[i][val3D_pred_mask]
        stats_mask0 = pad_mask0[i] & (gt_m0[i] != -2)
        stats_mask1 = pad_mask1[i] & (gt_m1[i] != -2)
        gt_stats0 = gt_m0[i][stats_mask0]
        pred_stats0 = pred_m0[i][stats_mask0]
        gt_stats1 = gt_m1[i][stats_mask1]
        pred_stats1 = pred_m1[i][stats_mask1]
        fn_stats0 = (gt_stats0 > -1) & (pred_stats0 == -1)
        tn_stats0 = (gt_stats0 == -1) & (pred_stats0 == -1)
        fn_stats1 = (gt_stats1 > -1) & (pred_stats1 == -1)
        tn_stats1 = (gt_stats1 == -1) & (pred_stats1 == -1)
        num_false_negatives0 = int(fn_stats0.sum())
        num_true_negatives0 = int(tn_stats0.sum())
        num_false_negatives1 = int(fn_stats1.sum())
        num_true_negatives1 = int(tn_stats1.sum())
        num_true_positives = int(true_positives.sum())
        num_false_positives = int((~true_positives).sum())

        # Filter all valid keypoints to get only positives
        kp0_pos_mask = pad_mask0[i] & (gt_m0[i] > -1)
        kp1_pos_mask = pad_mask1[i] & (gt_m1[i] > -1)
        plot_keypoints(
            [kp0[i][kp0_pos_mask], kp1[i][kp1_pos_mask]],
            axes=axes[0],
            facecolors=["limegreen", "limegreen"],
            edgecolors=["none", "none"],
            ps=[6, 6],
            lw=[0, 0],
        )

        # Filter all valid keypoint to get only hard negatives
        kp0_neg_mask =  pad_mask0[i] & (gt_m0[i] == -1)
        kp1_neg_mask = pad_mask1[i] & (gt_m1[i] == -1)
        plot_keypoints(
            [kp0[i][kp0_neg_mask], kp1[i][kp1_neg_mask]],
            axes=axes[0],
            facecolors=["blue", "blue"],
            edgecolors=["none", "none"],
            ps=[6, 6],
            lw=[0, 0],
        )

        # Filter all valid keypoint to get only ignored
        kp0_ign_mask = pad_mask0[i] & (gt_m0[i] == -2)
        kp1_ign_mask = pad_mask1[i] & (gt_m1[i] == -2)
        plot_keypoints(
            [kp1[i][kp0_ign_mask], kp1[i][kp1_ign_mask]],
            axes=axes[0],
            facecolors=["lightgray", "lightgray"],
            edgecolors=["none", "none"],
            ps=[1, 1],
            lw=[0, 0],
            a=[0.6, 0.6],
        )

        kpts0_stats = kp0[i][stats_mask0]
        kpts1_stats = kp1[i][stats_mask1]

        plot_keypoints(
            [kpts0_stats[fn_stats0], kpts1_stats[fn_stats1]],
            axes=axes[0],
            facecolors=["none", "none"],
            edgecolors=["black", "black"],
            ps=[5, 5],
            lw=[0.5, 0.5],
        )
        plot_keypoints(
            [kpts0_stats[tn_stats0], kpts1_stats[tn_stats1]],
            axes=axes[0],
            facecolors=["none", "none"],
            edgecolors=["blue", "blue"],
            ps=[6, 6],
            lw=[0.8, 0.8],
        )

        if kpm0.shape[0]:
            match_colors = cm_RdGn(true_positives)
            point_colors = []
            for g in gt_m0[i][val3D_pred_mask]:
                if g > -1:
                    point_colors.append("limegreen")
                elif g == -1:
                    point_colors.append("blue")
                else:
                    point_colors.append("lightgray")
            plot_matches(
                kpm0,
                kpm1,
                color=match_colors.tolist(),
                axes=axes[0],
                a=0.5,
                lw=1.0,
                ps=0.0,
            )
            plot_keypoints(
                [kpm0, kpm1],
                axes=axes[0],
                facecolors=[point_colors, point_colors],
                edgecolors=[match_colors.tolist(), match_colors.tolist()],
                ps=[8, 8],
                lw=[0.8, 0.8],
            )
        title_line0 = " | ".join(
            [   
                f"KP: tot/pos/neg/ign                                           "  #using tabs
                f"ov: {overlap[i]:.2f}",
                f"GT_POS: {(gt_m0[i]>-1).sum()}",

            ]
        )
        title_line1 = " | ".join(
            [
                f"IMG0-> "    
                f"KP: {pad_mask0[i].sum()}/{kp0_pos_mask.sum()}/{kp0_neg_mask.sum()}/{kp0_ign_mask.sum()}",
                f"KP_3D: {val3D_mask0[i].sum()}",
                f"TP(lines): {num_true_positives}",
                f"FP(lines): {num_false_positives}",
                f"TN0: {num_true_negatives0}",
                f"FN0: {num_false_negatives0}",
            ]
        )
        title_line2 = " | ".join(
            [
                f"IMG1-> " 
                f"KP: {pad_mask1[i].sum()}/{kp1_pos_mask.sum()}/{kp1_neg_mask.sum()}/{kp1_ign_mask.sum()}",
                f"KP_3D: {val3D_mask1[i].sum()}",
                f"TP(lines): {num_true_positives}",
                f"FP(lines): {num_false_positives}",
                f"TN1: {num_true_negatives1}",
                f"FN1: {num_false_negatives1}",
            ]
        )
        fig.suptitle(
            f"{title_line0}\n{title_line1}\n{title_line2}", fontsize=8, y=0.99, va="bottom"
        )
        fig.subplots_adjust(top=1.1)
        
        # Add colored legend in a column on the right side
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='limegreen', edgecolor='none', label='GT pos'),
            Patch(facecolor='blue', edgecolor='none', label='GT neg'),
            Patch(facecolor='lightgray', edgecolor='none', label='GT ign'),
            Patch(facecolor='none', edgecolor='limegreen', label='TP'),
            Patch(facecolor='none', edgecolor='red', label='FP'),
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
