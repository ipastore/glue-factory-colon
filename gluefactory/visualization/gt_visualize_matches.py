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

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred.get("gt_matches0", m0)

    figs = []
    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
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
            pad=0.1,
        )
        fig.set_size_inches(figsize[0], figsize[1])

        correct = gtm0[i][valid] == m0[i][valid]
        if kpm0.shape[0]:
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(correct).tolist(),
                axes=axes[0],
                a=0.5,
                lw=1.0,
                ps=0.0,
            )

        # Plot invalid (gt == -1) keypoints in red on the first image
        neg_mask = gtm0[i] == -1
        if neg_mask.any():
            neg0 = kp0[i][neg_mask].numpy()
            plot_keypoints([neg0, torch.zeros((0, 2))], axes=axes[0], colors=["red", "red"], ps=5)

        if "heatmap0" in pred.keys():
            plot_heatmaps(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ],
                axes=axes[0],
                a=1.0,
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            plot_heatmaps([view0["depth"][i], view1["depth"][i]], axes=axes[0], a=1.0)

        num_kp = (kp0[i].shape[0], kp1[i].shape[0])
        num_matches = kpm0.shape[0]
        num_invalid = int((gtm0[i] == -1).sum().item()) if gtm0 is not None else 0
        overlap = data.get("overlap_0to1", None)
        if overlap is not None:
            ov = overlap[i] if hasattr(overlap, "__len__") else overlap
            overlap_val = float(ov.detach().cpu().item()) if torch.is_tensor(ov) else float(ov)
        else:
            overlap_val = None
        stats_parts = [
            f"overlap: {overlap_val:.3f}" if overlap_val is not None else "overlap: n/a",
            f"kp: {num_kp[0]}/{num_kp[1]}",
            f"matches: {num_matches}",
            f"invalid: {num_invalid}",
        ]
        fig.suptitle(" | ".join(stats_parts), fontsize=6, y=0.99, va="bottom")
        figs.append(fig)

    return figs
