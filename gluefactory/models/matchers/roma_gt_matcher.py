from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ...geometry.gt_generation import (
    gt_matches_from_roma,
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


class RomaGTMatcher(BaseModel):
    default_conf = {
        #TODO add configs for roma
        "save_fig_when_debug": False,
    }

    def _init(self, conf):
        pass

    @AMP_CUSTOM_FWD_F32
    def _forward(self, data):
        gt = {}
        keys = [
            # "sparse_depth0",
            # "valid_3D_mask0",
            # "sparse_depth1",
            # "valid_3D_mask1",
            # "point3D_ids0",
            # "point3D_ids1",
            # "valid_depth_mask0",
            # "valid_depth_mask1"
        ]
        kw = {k: data[k] for k in keys}
        #TODO implement gt_matches_from_roma
        gt = gt_matches_from_roma(
            data["keypoints0"],
            data["keypoints1"],
            data,
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

                gt_pos_figs = make_gt_pos_figs(
                    gt,
                    data,
                    n_pairs=data["keypoints0"].shape[0],
                    pos_th=self.conf.th_positive,
                )
                pos_dir = base_dir / "seq_map_gt_pos"
                _save_figures(gt_pos_figs, names, pos_dir)
        return gt

    def loss(self, pred, data):
        raise NotImplementedError
