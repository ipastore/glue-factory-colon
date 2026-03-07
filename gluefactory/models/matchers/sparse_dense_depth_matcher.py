from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ...geometry.gt_generation import (
    gt_matches_from_pose_sparse_dense_map,
)
from ... import settings
from ..base_model import BaseModel
from ...visualization.gt_visualize_matches import (
    make_gt_pos_neg_ign_figs,
    make_gt_pos_figs,
)

# Hacky workaround for torch.amp.custom_fwd to support older versions of PyTorch.
AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


def _prepare_fig_names(names, num_figs):
    """Format the filenames used when saving debug figures."""
    if isinstance(names, torch.Tensor):
        names = names.tolist() if names.ndim > 0 else names.item()
    if hasattr(names, "item"):
        names = names.item()

    formatted = []
    for idx in range(num_figs):
        fname = names
        if isinstance(names, (list, tuple)):
            fname = names[idx] if idx < len(names) else names[0]
        fname = str(fname).replace("/", "__")
        parts = fname.split("__")
        if len(parts) >= 3:
            fname = f"{parts[0]}_{parts[1]}_{parts[2]}"
        elif len(parts) == 2:
            fname = f"{parts[0]}_{parts[1]}"
        else:
            fname = parts[0]
        formatted.append(fname)
    return formatted


def _save_figures(figs, names, save_dir):
    """Persist a list of matplotlib figures to disk and close them."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname, fig in zip(names, figs):
        fig.savefig(
            save_dir / f"{fname}.png", bbox_inches="tight", pad_inches=0, dpi=300
        )
        plt.close(fig)


def _build_sfm_plot_gt(data):
    if "point3D_ids0" not in data or "point3D_ids1" not in data:
        return None
    ids0 = data["point3D_ids0"].long()
    ids1 = data["point3D_ids1"].long()
    valid0 = ids0 >= 0
    valid1 = ids1 >= 0
    same = (ids0.unsqueeze(-1) == ids1.unsqueeze(-2)) & valid0.unsqueeze(
        -1
    ) & valid1.unsqueeze(-2)
    has0 = same.any(-1)
    has1 = same.any(-2)
    idx0 = same.float().argmax(-1)
    idx1 = same.float().argmax(-2)

    m0 = torch.full_like(ids0, -2, dtype=torch.long)
    m1 = torch.full_like(ids1, -2, dtype=torch.long)
    m0 = torch.where(has0, idx0, m0)
    m1 = torch.where(has1, idx1, m1)
    z0 = torch.zeros_like(has0)
    z1 = torch.zeros_like(has1)
    return {
        "matches0": m0,
        "matches1": m1,
        "mask_pos_3d_map0": has0,
        "mask_pos_3d_map1": has1,
        "mask_pos_reproj0": z0,
        "mask_pos_reproj1": z1,
    }


class SparseDenseDepthMatcher(BaseModel):
    default_conf = {
        "use_gt_pos_for_plot": False,
        "use_gt_pos": False,  # legacy alias, plotting-only
        "th_positive": 3.0,
        "th_negative": 5.0,
        "th_epi": None,  # add some more epi outliers
        "th_consistency": None,  # check for projection consistency in px
        "save_fig_when_debug": False,
    }

    def _init(self, conf):
        pass

    @AMP_CUSTOM_FWD_F32
    def _forward(self, data):
        gt = {}
        keys = [
                "depth_keypoints0",
                "depth_keypoints1",
                "valid_depth_keypoints0",
                "valid_depth_keypoints1"

        ]
        kw = {k: data.get(k) for k in keys}
        gt = gt_matches_from_pose_sparse_dense_map(
            data["keypoints0"],
            data["keypoints1"],
            data,
            pos_th=self.conf.th_positive,
            neg_th=self.conf.th_negative,
            epi_th=self.conf.th_epi,
            cc_th=self.conf.th_consistency,
            **kw,
        )
        if self.conf.save_fig_when_debug:
            if "image" in data["view0"] and "image" in data["view1"]:

                figs = make_gt_pos_neg_ign_figs(
                    gt,
                    data,
                    n_pairs=data["keypoints0"].shape[0],
                    pos_th=self.conf.th_positive,
                    neg_th=self.conf.th_negative,
                )
                base_dir = Path(settings.TRAINING_PATH) / getattr(
                    self.conf, "experiment_name", "debug"
                )
                names = _prepare_fig_names(
                    data.get("names", data.get("idx", "pair")),
                    num_figs=data["keypoints0"].shape[0],
                )
                save_dir = base_dir / "seq_map_gt_viz"
                _save_figures(figs, names, save_dir)

                gt_pos_reproj_figs = make_gt_pos_figs(
                    gt,
                    data,
                    n_pairs=data["keypoints0"].shape[0],
                    pos_th=self.conf.th_positive,
                )
                pos_reproj_dir = base_dir / "seq_map_gt_pos_reproj"
                _save_figures(gt_pos_reproj_figs, names, pos_reproj_dir)

                use_gt_pos_for_plot = self.conf.use_gt_pos_for_plot or self.conf.use_gt_pos
                if use_gt_pos_for_plot:
                    gt_sfm = _build_sfm_plot_gt(data)
                    if gt_sfm is not None:
                        gt_pos_sfm_figs = make_gt_pos_figs(
                            gt_sfm,
                            data,
                            n_pairs=data["keypoints0"].shape[0],
                            pos_th=self.conf.th_positive,
                        )
                        pos_sfm_dir = base_dir / "seq_map_gt_pos_sfm"
                        _save_figures(gt_pos_sfm_figs, names, pos_sfm_dir)
        return gt

    def loss(self, pred, data):
        raise NotImplementedError
