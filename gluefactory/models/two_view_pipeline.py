"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

import math
import time

import torch
from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel
from .utils.misc import mask_with_mode

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
        "keypoint_rotation": {
            "enabled": False,
            "max_deg": 180.0,
            "view": 0,
            "train_only": True,
        },
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def _metric_tensor(self, batch_size, device, value):
        return torch.full(
            (batch_size,), float(value), device=device, dtype=torch.float32
        )

    def _profile_call(self, device, fn):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated_peak = torch.cuda.max_memory_allocated(device)
            reserved_peak = torch.cuda.max_memory_reserved(device)
            memory_mb = max(allocated_peak - baseline, 0) / (1024**2)
            allocated_peak_mb = allocated_peak / (1024**2)
            reserved_peak_mb = reserved_peak / (1024**2)
        else:
            memory_mb = None
            allocated_peak_mb = None
            reserved_peak_mb = None
        return (
            output,
            (time.perf_counter() - start) * 1e3,
            memory_mb,
            allocated_peak_mb,
            reserved_peak_mb,
        )

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        extract_time_ms = None
        extract_core_time_ms = None
        extract_memory_mb = None
        extract_allocated_peak_mb = None
        extract_reserved_peak_mb = None
        device = data_i["image"].device
        if self.conf.extractor.name and not skip_extract:
            extractor_fn = lambda: self.extractor(data_i)
            (
                extractor_pred,
                extract_time_ms,
                extract_memory_mb,
                extract_allocated_peak_mb,
                extract_reserved_peak_mb,
            ) = self._profile_call(device, extractor_fn)
            if "extractor_core_time_ms" in extractor_pred:
                extract_core_time_ms = extractor_pred.pop("extractor_core_time_ms")
            pred_i = {**pred_i, **extractor_pred}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            extractor_input = {**data_i, **pred_i}
            extractor_fn = lambda: self.extractor(extractor_input)
            (
                extractor_pred,
                extract_time_ms,
                extract_memory_mb,
                extract_allocated_peak_mb,
                extract_reserved_peak_mb,
            ) = self._profile_call(device, extractor_fn)
            if "extractor_core_time_ms" in extractor_pred:
                extract_core_time_ms = extractor_pred.pop("extractor_core_time_ms")
            pred_i = {**pred_i, **extractor_pred}
        return (
            pred_i,
            extract_time_ms,
            extract_core_time_ms,
            extract_memory_mb,
            extract_allocated_peak_mb,
            extract_reserved_peak_mb,
        )

    def _rotate_keypoints(self, kpts, center, angles):
        rel = kpts - center[:, None, :]
        cos = torch.cos(angles)[:, None]
        sin = torch.sin(angles)[:, None]
        rot_kpts = torch.stack(
            [
                cos * rel[..., 0] - sin * rel[..., 1],
                sin * rel[..., 0] + cos * rel[..., 1],
            ],
            dim=-1,
        )
        return rot_kpts + center[:, None, :]

    def _apply_keypoint_rotation(self, pred, data):
        rot_conf = self.conf.keypoint_rotation
        if not rot_conf.enabled:
            return pred
        if rot_conf.train_only and not self.training:
            return pred
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            return pred

        view_idx = int(rot_conf.view)
        view_key = f"view{view_idx}"
        kpts_key = f"keypoints{view_idx}"
        oris_key = f"oris{view_idx}"
        scores_key = f"keypoint_scores{view_idx}"

        if kpts_key not in pred:
            return pred

        kpts = pred[kpts_key]
        device = kpts.device
        batch_size = kpts.shape[0]
        max_rad = float(rot_conf.max_deg) * math.pi / 180.0
        if max_rad <= 0:
            return pred

        angles = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * max_rad
        pred[f"rot_angle{view_idx}"] = angles
        camera = data[view_key]["camera"].to(device)
        center = camera.c
        rot_kpts = self._rotate_keypoints(kpts, center, angles)

        rot_oris = None
        if oris_key in pred:
            oris = pred[oris_key]
            if oris.dim() == 3:
                rot_oris = oris + angles[:, None, None]
            else:
                rot_oris = oris + angles[:, None]
            rot_oris = torch.atan2(torch.sin(rot_oris), torch.cos(rot_oris))

        keep = camera.in_image(rot_kpts)

        image_size = data[view_key].get("image_size", camera.size).to(device)
        bounds = (
            torch.zeros_like(image_size)[:, None, :],
            (image_size - 1.0)[:, None, :],
        )
        pred[kpts_key] = mask_with_mode(
            rot_kpts, keep, mode="random_c", bounds=bounds
        )

        if rot_oris is not None:
            pred[oris_key] = mask_with_mode(rot_oris, keep, mode="zeros")

        desc_key = f"descriptors{view_idx}"
        if desc_key in pred:
            pred[desc_key] = mask_with_mode(pred[desc_key], keep, mode="random")

        scales_key = f"scales{view_idx}"
        if scales_key in pred:
            pred[scales_key] = mask_with_mode(pred[scales_key], keep, mode="zeros")

        if scores_key in pred:
            pred[scores_key] = mask_with_mode(pred[scores_key], keep, mode="zeros")

        sparse_key = f"sparse_depth{view_idx}"
        if sparse_key in pred:
            pred[sparse_key] = mask_with_mode(pred[sparse_key], keep, mode="minus_one")

        ids_key = f"point3D_ids{view_idx}"
        if ids_key in pred:
            pred[ids_key] = mask_with_mode(pred[ids_key], keep, mode="minus_one")

        valid_depth_key = f"valid_depth_mask{view_idx}"
        if valid_depth_key in pred:
            pred[valid_depth_key] = mask_with_mode(
                pred[valid_depth_key], keep, mode=False
            )

        valid_3d_key = f"valid_3D_mask{view_idx}"
        if valid_3d_key in pred:
            pred[valid_3d_key] = mask_with_mode(pred[valid_3d_key], keep, mode=False)

        ignore_val = -2
        if "gt_matches0" in pred and view_idx == 0:
            pred["gt_matches0"] = pred["gt_matches0"].masked_fill(~keep, ignore_val)
            pred["gt_matching_scores0"] = (pred["gt_matches0"] > -1).float()
            if "gt_matches1" in pred:
                gt_matches1 = pred["gt_matches1"].clone()
                valid_m1 = gt_matches1 >= 0
                mapped = gt_matches1.clamp(min=0).long()
                invalid_mapped = (~keep).gather(1, mapped)
                mask = valid_m1 & invalid_mapped
                gt_matches1 = torch.where(
                    mask, gt_matches1.new_full((), ignore_val), gt_matches1
                )
                pred["gt_matches1"] = gt_matches1
                pred["gt_matching_scores1"] = (gt_matches1 > -1).float()

            for key in (
                "gt_assignment",
                "gt_reward",
                "gt_depth_keypoints0",
                "gt_proj_0to1",
                "gt_visible0",
                "gt_mask_pos_3d_map0",
                "gt_mask_pos_reproj0",
                "gt_mask_neg_reproj0",
                "gt_mask_neg_epi0",
            ):
                if key not in pred:
                    continue
                val = pred[key]
                mode = False if val.dtype == torch.bool else "zeros"
                pred[key] = mask_with_mode(val, keep, mode=mode)
        return pred

    def _forward(self, data):
        device = data["view0"]["image"].device
        batch_size = data["view0"]["image"].shape[0]
        (
            pred0,
            extract_time0,
            extract_core_time0,
            extract_memory0,
            extract_allocated0,
            extract_reserved0,
        ) = self.extract_view(data, "0")
        (
            pred1,
            extract_time1,
            extract_core_time1,
            extract_memory1,
            extract_allocated1,
            extract_reserved1,
        ) = self.extract_view(data, "1")
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }
        extractor_time_ms = sum(
            t for t in [extract_time0, extract_time1] if t is not None
        )
        has_extractor_time = extract_time0 is not None or extract_time1 is not None
        extractor_core_time_ms = None
        if extract_core_time0 is not None or extract_core_time1 is not None:
            extractor_core_time_ms = 0
            for t in [extract_core_time0, extract_core_time1]:
                if t is not None:
                    extractor_core_time_ms += t
        extractor_memory_mb = sum(
            m for m in [extract_memory0, extract_memory1] if m is not None
        )
        has_extractor_memory = (
            extract_memory0 is not None or extract_memory1 is not None
        )

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        pred = self._apply_keypoint_rotation(pred, data) #Only for  sparse_endomapper (maybe to be erased)
        matcher_time_ms = None
        matcher_memory_mb = None
        matcher_allocated_peak_mb = None
        matcher_reserved_peak_mb = None
        if self.conf.matcher.name:
            matcher_fn = lambda: self.matcher({**data, **pred})
            (
                matcher_pred,
                matcher_time_ms,
                matcher_memory_mb,
                matcher_allocated_peak_mb,
                matcher_reserved_peak_mb,
            ) = self._profile_call(device, matcher_fn)
            pred = {**pred, **matcher_pred}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}
        forward_allocated_peaks = [
            extract_allocated0,
            extract_allocated1,
            matcher_allocated_peak_mb,
        ]
        forward_reserved_peaks = [
            extract_reserved0,
            extract_reserved1,
            matcher_reserved_peak_mb,
        ]
        if device.type == "cuda":
            forward_allocated_memory_mb = max(
                peak for peak in forward_allocated_peaks if peak is not None
            )
            forward_reserved_memory_mb = max(
                peak for peak in forward_reserved_peaks if peak is not None
            )
        else:
            forward_allocated_memory_mb = None
            forward_reserved_memory_mb = None
        image0 = data["view0"]["image"]
        image1 = data["view1"]["image"]
        pair_resolution = image0.shape[-2] * image0.shape[-1]
        pair_resolution += image1.shape[-2] * image1.shape[-1]
        if has_extractor_time:
            pred["extractor_time_ms"] = self._metric_tensor(
                batch_size, device, extractor_time_ms
            )
            if extractor_core_time_ms is not None:
                pred["extractor_core_time_ms"] = extractor_core_time_ms
            total_time_ms = extractor_time_ms
            if matcher_time_ms is not None:
                pred["matcher_time_ms"] = self._metric_tensor(
                    batch_size, device, matcher_time_ms
                )
                total_time_ms += matcher_time_ms
            pred["total_time_ms"] = self._metric_tensor(
                batch_size, device, total_time_ms
            )
        elif matcher_time_ms is not None:
            pred["total_time_ms"] = self._metric_tensor(
                batch_size, device, matcher_time_ms
            )
        if has_extractor_memory:
            pred["extractor_memory_mb"] = self._metric_tensor(
                batch_size, device, extractor_memory_mb
            )
            if matcher_memory_mb is not None:
                pred["matcher_memory_mb"] = self._metric_tensor(
                    batch_size, device, matcher_memory_mb
                )
        elif matcher_memory_mb is not None:
            pred["matcher_memory_mb"] = self._metric_tensor(
                batch_size, device, matcher_memory_mb
            )
        if forward_allocated_memory_mb is not None:
            pred["forward_allocated_memory_mb"] = self._metric_tensor(
                batch_size, device, forward_allocated_memory_mb
            )
            pred["forward_reserved_memory_mb"] = self._metric_tensor(
                batch_size, device, forward_reserved_memory_mb
            )
        pred["pair_resolution"] = self._metric_tensor(
            batch_size, device, pair_resolution
        )
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
