from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .roma import RoMa
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
        "save_fig_when_debug": False,
        "th_positive": None,
        "th_negative": None,
        "roma": {
            "name": "matchers.roma",
            "weights": "indoor",
            "upsample_preds": True,
            "symmetric": True,
            "internal_hw": (560, 560),
            "output_hw": None,
            "sample": False,
            "mixed_precision": True,
            "add_cycle_error": False,
            "sample_num_matches": 0,
            "sample_mode": "threshold_balanced",
            "filter_threshold": 0.05,
            "max_kp_error": 2.0,
            "mutual_check": True,
        },
    }
    required_data_keys = ["view0", "view1", "keypoints0", "keypoints1"]

    def _init(self, conf):
        self.roma = RoMa(conf.roma)

    @AMP_CUSTOM_FWD_F32
    def _forward(self, data):
        roma_pred = self.roma(data)
        valid0 = data.get("keypoint_scores0")
        valid1 = data.get("keypoint_scores1")
        if valid0 is None or valid1 is None:
            raise ValueError(
                "RomaGTMatcher requires keypoint_scores0/1 to mask padded features."
            )
        valid0 = valid0 > 0
        valid1 = valid1 > 0
        gt = gt_matches_from_roma(
            data["keypoints0"],
            data["keypoints1"],
            data,
            matches0=roma_pred["matches0"],
            matches1=roma_pred["matches1"],
            matching_scores0=roma_pred.get("matching_scores0"),
            matching_scores1=roma_pred.get("matching_scores1"),
            valid0=valid0,
            valid1=valid1,
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
