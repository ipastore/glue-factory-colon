import torch

from ...geometry.gt_generation import (
    gt_matches_from_pose_sparse_depth,
    gt_matches_from_pose_sparse_map,
)
from ..base_model import BaseModel

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
        return result

    def loss(self, pred, data):
        raise NotImplementedError
