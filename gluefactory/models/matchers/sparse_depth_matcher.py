import matplotlib.pyplot as plt
import torch
from pathlib import Path

from ...geometry.gt_generation import (
    gt_matches_from_pose_sparse_depth,
    gt_matches_from_pose_sparse_map,
)
from ... import settings
from ..base_model import BaseModel
from ...visualization.gt_visualize_matches import make_gt_debug_figures

# Hacky workaround for torch.amp.custom_fwd to support older versions of PyTorch.
AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


class SparseDepthMatcher(BaseModel):
    default_conf = {
        "use_sparse_depth": True,
        "th_positive": 3.0,
        "th_negative": 5.0,
        "th_epi": None,  # add some more epi outliers
        "th_consistency": None,  # check for projection consistency in px
        "save_fig_when_debug": False,
    }

    required_data_keys = [
        "view0",
        "view1",
        "T_0to1",
        "keypoints0",
        "keypoints1",
        "sparse_depth0",
        "sparse_depth1",
        "valid_3D_mask0",
        "valid_3D_mask1",
        "point3D_ids0",
        "point3D_ids1",
    ]

    def _init(self, conf):
        pass

    @AMP_CUSTOM_FWD_F32
    def _forward(self, data):
        result = {}
        if self.conf.use_sparse_depth:
            keys = [
                "sparse_depth0",
                "valid_3D_mask0",
                "sparse_depth1",
                "valid_3D_mask1",
            ]
            kw = {k: data[k] for k in keys}
            result = gt_matches_from_pose_sparse_depth(
                data["keypoints0"],
                data["keypoints1"],
                data,
                pos_th=self.conf.th_positive,
                neg_th=self.conf.th_negative,
                epi_th=self.conf.th_epi,
                cc_th=self.conf.th_consistency,
                **kw,
            )
        else:
            keys = [
                "point3D_ids0",
                "valid_3D_mask0",
                "point3D_ids1",
                "valid_3D_mask1",
            ]
            kw = {k: data[k] for k in keys}
            result = gt_matches_from_pose_sparse_map(
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
                with torch.no_grad():
                    plot_pred = {
                        **result,
                        "keypoints0": data["keypoints0"],
                        "keypoints1": data["keypoints1"],
                        "gt_matches0": result["matches0"],
                    }
                figs = make_gt_debug_figures(
                    plot_pred, data, n_pairs=data["keypoints0"].shape[0]
                )
                base_dir = Path(settings.TRAINING_PATH) / getattr(
                    self.conf, "experiment_name", "debug"
                )
                save_dir = base_dir / "seq_map_gt_viz"
                save_dir.mkdir(parents=True, exist_ok=True)
                names = data.get("names", data.get("idx", "pair"))
                if hasattr(names, "item"):
                    names = names.item()
                for j, fig in enumerate(figs):
                    fname = names
                    if isinstance(names, (list, tuple)):
                        fname = names[j] if j < len(names) else names[0]
                    fname = str(fname).replace("/", "__")
                    parts = fname.split("__")
                    if len(parts) >= 3:
                        fname = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    elif len(parts) == 2:
                        fname = f"{parts[0]}_{parts[1]}"
                    else:
                        fname = parts[0]
                    fig.savefig(save_dir / f"{fname}.png", bbox_inches="tight", pad_inches=0, dpi=300)
                    plt.close(fig)
        return result

    def loss(self, pred, data):
        raise NotImplementedError
