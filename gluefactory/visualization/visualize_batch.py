import torch
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches

import logging
import warnings
logger = logging.getLogger(__name__)

def make_match_figures(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kp0_i = kp0[i]
        kp1_i = kp1[i]

        logger.debug("kp0: %d kp1: %d; BEFORE valid_depth", kp0_i.shape[0], kp1_i.shape[0])

        if "valid_depth_keypoints0" in pred and pred["valid_depth_keypoints0"] is not None:
            mask0 = pred["valid_depth_keypoints0"][i].bool()
            before0 = kp0_i.shape[0]
            kp0_i = kp0_i[mask0]
            logger.debug(
                "valid_depth0: mask_len=%d true=%d kept=%d removed=%d",
                mask0.numel(), int(mask0.sum()), kp0_i.shape[0], before0 - kp0_i.shape[0]
            )

        if "valid_depth_keypoints1" in pred and pred["valid_depth_keypoints1"] is not None:
            mask1 = pred["valid_depth_keypoints1"][i].bool()
            before1 = kp1_i.shape[0]
            kp1_i = kp1_i[mask1]
            logger.debug(
                "valid_depth1: mask_len=%d true=%d kept=%d removed=%d",
                mask1.numel(), int(mask1.sum()), kp1_i.shape[0], before1 - kp1_i.shape[0]
            )

        logger.debug("kp0: %d kp1: %d; AFTER valid_depth", kp0_i.shape[0], kp1_i.shape[0])
        
        kpts.append([kp0_i, kp1_i])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    # #### DEBUG FOR SEE KPTS ####
    # # Save plotted samples (grid + full-res source images + keypoint report).
    # try:
    #     def plot_padded_crosses(ax, kp, alpha=1, size=22):
    #         if isinstance(kp, torch.Tensor):
    #             kp = kp.detach().cpu().numpy()
    #         if kp is None or len(kp) == 0:
    #             return
    #         kp = np.asarray(kp)
    #         ax.scatter(
    #             kp[:, 0],
    #             kp[:, 1],
    #             s=size,
    #             marker="x",
    #             c="red",
    #             alpha=alpha,
    #             linewidths=1.0,
    #         )

    #     pair_names = data.get("name", [])
    #     view0_names = data["view0"].get("name", [])
    #     view1_names = data["view1"].get("name", [])

    #     if not isinstance(pair_names, (list, tuple)):
    #         pair_names = [pair_names] * n_pairs
    #     if not isinstance(view0_names, (list, tuple)):
    #         view0_names = [view0_names] * n_pairs
    #     if not isinstance(view1_names, (list, tuple)):
    #         view1_names = [view1_names] * n_pairs

    #     pair_name0 = str(pair_names[0] if len(pair_names) > 0 else "pair_0")
    #     if "/" in pair_name0:
    #         seq, pair_id0 = pair_name0.split("/", 1)
    #     else:
    #         seq, pair_id0 = "unknown", pair_name0
    #     safe_pair0 = pair_id0.replace("/", "_")

    #     fig_dir = Path("outputs/debug_kpts") / seq
    #     fig_dir.mkdir(parents=True, exist_ok=True)
    #     fig_path = fig_dir / f"{safe_pair0}_grid.png"
    #     fig.savefig(fig_path, bbox_inches="tight", pad_inches=0)

    #     detected0 = pred.get("num_keypoints_detected0")
    #     detected1 = pred.get("num_keypoints_detected1")
    #     padded0 = pred.get("num_keypoints_padded0")
    #     padded1 = pred.get("num_keypoints_padded1")

    #     for i in range(n_pairs):
    #         pair_name = str(pair_names[i] if i < len(pair_names) else f"pair_{i}")
    #         if "/" in pair_name:
    #             _, pair_id = pair_name.split("/", 1)
    #         else:
    #             pair_id = pair_name
    #         safe_pair = pair_id.replace("/", "_")
    #         name0 = str(view0_names[i] if i < len(view0_names) else f"view0_{i}.png")
    #         name1 = str(view1_names[i] if i < len(view1_names) else f"view1_{i}.png")
    #         name0 = Path(name0).name
    #         name1 = Path(name1).name

    #         img0 = view0["image"][i].permute(1, 2, 0).numpy()
    #         img1 = view1["image"][i].permute(1, 2, 0).numpy()
    #         if img0.shape[-1] == 1:
    #             img0 = img0[..., 0]
    #         if img1.shape[-1] == 1:
    #             img1 = img1[..., 0]

    #         d0 = int(detected0[i].item()) if detected0 is not None else kp0[i].shape[0]
    #         d1 = int(detected1[i].item()) if detected1 is not None else kp1[i].shape[0]

    #         out0 = fig_dir / name0
    #         out1 = fig_dir / name1
    #         for img, kp, det_n, out in (
    #             (img0, kp0[i], d0, out0),
    #             (img1, kp1[i], d1, out1),
    #         ):
    #             h, w = img.shape[:2]
    #             dpi = 100
    #             fig_img = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    #             ax = fig_img.add_axes([0, 0, 1, 1])
    #             ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    #             ax.set_xlim([0, w])
    #             ax.set_ylim([h, 0])
    #             ax.set_axis_off()
    #             if det_n > 0:
    #                 plot_keypoints(
    #                     [kp[:det_n]],
    #                     axes=[ax],
    #                     colors="limegreen",
    #                     a=0.6,
    #                     ps=12,
    #                 )
    #             if det_n < kp.shape[0]:
    #                 plot_padded_crosses(ax, kp[det_n:], alpha=1, size=24)
    #             fig_img.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0)
    #             plt.close(fig_img)

    #         p0 = int(padded0[i].item()) if padded0 is not None else -1
    #         p1 = int(padded1[i].item()) if padded1 is not None else -1

    #         report_path = fig_dir / f"{safe_pair}_kpts_report.txt"
    #         with open(report_path, "w", encoding="utf-8") as f:
    #             f.write("image_name,detected_keypoints,padded_keypoints\n")
    #             f.write(f"{name0},{d0},{p0}\n")
    #             f.write(f"{name1},{d1},{p1}\n")
    # except Exception:
    #     pass
    # #### DEBUG FOR SEE KPTS ####

    return {"matching": fig}
