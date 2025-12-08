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
    m0 = data["matches0"]
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
        num_invalid = int((~correct).sum())
        kpts_pair = [kp0[i], kp1[i]]
        plot_keypoints(kpts_pair, axes=axes[0], colors="royalblue")
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

        num_kp = (kp0[i].shape[0], kp1[i].shape[0])
        num_matches = kpm0.shape[0]
        num_predictions =  int((m0[i]>-1).sum())
        num_matches_gt = int((gtm0[i]>-1).sum()) 
        overlap = data.get("overlap_0to1", None)
        if overlap is not None:
            ov = overlap[i] if hasattr(overlap, "__len__") else overlap
            overlap_val = float(ov.detach().cpu().item()) if torch.is_tensor(ov) else float(ov)
        else:
            overlap_val = None
        stats_parts = [
            f"overlap: {overlap_val:.3f}" if overlap_val is not None else "overlap: n/a",
            f"kp: {num_kp[0]}/{num_kp[1]}",
            f"valid: {num_matches}",
            f"invalid: {num_invalid}",
            f"total_predictions: {num_predictions}",
            f"total_gt_matches: {num_matches_gt}"
        ]
        fig.suptitle(" | ".join(stats_parts), fontsize=6, y=0.99, va="bottom")
        figs.append(fig)

    return figs
