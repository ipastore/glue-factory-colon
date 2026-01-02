import math

import torch
import torch.nn.functional as F

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_image_grid, plot_keypoints, plot_matches


def _rotate_image_tensor(image, angle, center):
    _, h, w = image.shape
    device = image.device
    dtype = image.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)
    center = center.to(device=device, dtype=dtype)
    rel = grid - center
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    src = torch.stack(
        [cos * rel[..., 0] + sin * rel[..., 1], -sin * rel[..., 0] + cos * rel[..., 1]],
        dim=-1,
    )
    src = src + center
    denom_w = max(w - 1, 1)
    denom_h = max(h - 1, 1)
    src_x = 2.0 * (src[..., 0] / denom_w) - 1.0
    src_y = 2.0 * (src[..., 1] / denom_h) - 1.0
    grid_norm = torch.stack([src_x, src_y], dim=-1)
    warped = F.grid_sample(
        image[None],
        grid_norm[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped[0]


def make_rotation_compare_figures(pred_rot, pred_orig, data_, n_pairs=2, rot_view=0):
    pred_rot = pred_rot["0to1"] if "0to1" in pred_rot else pred_rot
    pred_orig = pred_orig["0to1"] if "0to1" in pred_orig else pred_orig
    pred_rot = batch_to_device(pred_rot, "cpu", non_blocking=False)
    pred_orig = batch_to_device(pred_orig, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]
    n_pairs = min(n_pairs, view0["image"].shape[0])
    figs = []
    rot_angles = pred_rot.get(f"rot_angle{rot_view}")
    cam = data[f"view{rot_view}"]["camera"]
    for i in range(n_pairs):
        imgs_orig = [
            view0["image"][i].permute(1, 2, 0),
            view1["image"][i].permute(1, 2, 0),
        ]
        imgs_rot = list(imgs_orig)
        if rot_angles is not None:
            angle = rot_angles[i]
            center = cam.c[i]
            if rot_view == 0:
                imgs_rot[0] = _rotate_image_tensor(
                    view0["image"][i], angle, center
                ).permute(
                    1, 2, 0
                )
            else:
                imgs_rot[1] = _rotate_image_tensor(
                    view1["image"][i], angle, center
                ).permute(
                    1, 2, 0
                )
        fig, axes = plot_image_grid(
            [imgs_rot, imgs_orig], return_fig=True, set_lim=True, pad=0.9
        )
        fig.subplots_adjust(hspace=0.1)

        for row, pred in enumerate((pred_rot, pred_orig)):
            kp0, kp1 = pred["keypoints0"][i], pred["keypoints1"][i]
            matches = pred["matches0"][i]
            gt_matches0 = pred.get("gt_matches0")
            if gt_matches0 is None:
                valid = matches > -1
                colors = "royalblue"
                pred_total = int(valid.sum().item())
                used = pred_total
                ok = None
                bad = None
            else:
                valid = (matches > -1) & (gt_matches0[i] >= -1)
                correct = gt_matches0[i][valid] == matches[valid]
                colors = cm_RdGn(correct).tolist()
                pred_total = int((matches > -1).sum().item())
                used = int(valid.sum().item())
                ok = int(correct.sum().item())
                bad = used - ok
            kpm0 = kp0[valid]
            kpm1 = kp1[matches[valid]]
            plot_keypoints([kp0, kp1], axes=axes[row], colors="royalblue")
            plot_matches(
                kpm0, kpm1, color=colors, axes=axes[row], a=0.5, lw=1.0, ps=0.0
            )
            label = "rotated" if row == 0 else "original"
            if row == 0 and rot_angles is not None:
                rot_deg = rot_angles[i].item() * 180.0 / math.pi
                label = f"{label} | rot {rot_deg:.1f}deg"
            if ok is None:
                title = f"{label} | matches {used}"
            else:
                title = f"{label} | ok/bad {ok}/{bad} | used {used}/{pred_total}"
            axes[row][0].set_title(title, fontsize=8, loc="left")
        figs.append(fig)
    return figs
