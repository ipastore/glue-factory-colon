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
    make_gt_roma_certainty_heatmap_figs,
    make_gt_roma_cycle_error_figs,
    make_gt_roma_demo_figs,
    make_gt_roma_matches_certainty_figs,
    make_gt_roma_matches_certainty_intersection_figs,
    make_gt_roma_matches_cycle_error_figs,
    make_gt_roma_matches_cycle_error_intersection_figs,
    make_gt_roma_raw_figs,
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


def _save_named_figure(fig, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def _tensor_stats(values, percentiles=()):
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            **{f"p{p}": float("nan") for p in percentiles},
        }
    values = values.detach().float().reshape(-1).cpu()
    stats = {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }
    if percentiles:
        q = torch.tensor(percentiles, dtype=values.dtype) / 100.0
        pq = torch.quantile(values, q)
        for idx, p in enumerate(percentiles):
            stats[f"p{p}"] = float(pq[idx].item())
    return stats


def _format_stats_block(label, stats, suffix=""):
    lines = [f"[{label}]"]
    for key in ["count", "mean", "median", "min", "max", "p90", "p95"]:
        if key in stats:
            value = stats[key]
            if key == "count":
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value:.4f}{suffix}")
    return lines


def _build_roma_pair_report(idx, name, roma_pred, gt, data, conf):
    image0 = data["view0"]["image"][idx]
    image1 = data["view1"]["image"][idx]
    overlap = data.get("overlap_0to1")
    overlap_val = overlap[idx].item() if overlap is not None else float("nan")
    thresholds = (0.05, 0.1, 0.2, 0.5)
    lines = [
        f"name: {name}",
        f"overlap_0to1: {overlap_val:.4f}",
        f"image0_shape_chw: {tuple(image0.shape)}",
        f"image1_shape_chw: {tuple(image1.shape)}",
        "[roma_conf]",
        f"filter_threshold: {conf.roma.filter_threshold}",
        f"max_kp_error: {conf.roma.max_kp_error}",
        f"mutual_check: {bool(conf.roma.mutual_check)}",
        f"symmetric: {bool(conf.roma.symmetric)}",
        f"add_cycle_error: {bool(conf.roma.add_cycle_error)}",
    ]

    for suffix in ("0", "1"):
        certainty = roma_pred[f"certainty{suffix}"][idx]
        certainty_stats = _tensor_stats(certainty)
        lines.extend(_format_stats_block(f"certainty{suffix}", certainty_stats))
        for thr in thresholds:
            coverage = float((certainty > thr).float().mean().item())
            lines.append(f"certainty{suffix}_coverage_gt_{thr:.2f}: {coverage:.4f}")

    for suffix in ("0", "1"):
        matches = gt[f"matches{suffix}"][idx]
        scores = gt[f"matching_scores{suffix}"][idx]
        valid = matches > -1
        match_stats = _tensor_stats(scores[valid])
        lines.extend(
            _format_stats_block(f"accepted_matches{suffix}_certainty", match_stats)
        )
        lines.append(f"accepted_matches{suffix}: {int(valid.sum().item())}")

    if "cycle_error0" in roma_pred and "cycle_error1" in roma_pred:
        for suffix in ("0", "1"):
            cycle = roma_pred[f"cycle_error{suffix}"][idx]
            cycle_stats = _tensor_stats(cycle, percentiles=(90, 95))
            lines.extend(
                _format_stats_block(
                    f"cycle_error{suffix}", cycle_stats, suffix=" px"
                )
            )
            for thr in (1.0, 2.0, 4.0):
                frac = float((cycle < thr).float().mean().item())
                lines.append(f"cycle_error{suffix}_coverage_lt_{thr:.1f}px: {frac:.4f}")

    return "\n".join(lines) + "\n"


def _save_roma_phase1_debug_outputs(roma_pred, gt, data, conf):
    base_dir = Path(settings.TRAINING_PATH) / getattr(conf, "experiment_name", "debug")
    names = _prepare_fig_names(
        data.get("names", data.get("idx", "pair")),
        num_figs=data["keypoints0"].shape[0],
    )
    n_pairs = data["keypoints0"].shape[0]
    raw_figs = make_gt_roma_raw_figs(roma_pred, data, n_pairs=n_pairs)
    pure_warp_figs, directional_figs = make_gt_roma_demo_figs(roma_pred, data, n_pairs=n_pairs)
    certainty_figs = make_gt_roma_certainty_heatmap_figs(
        roma_pred, data, n_pairs=n_pairs
    )
    matches_figs = make_gt_roma_matches_certainty_figs(gt, data, n_pairs=n_pairs)
    matches_intersection_figs = make_gt_roma_matches_certainty_intersection_figs(
        gt, data, n_pairs=n_pairs
    )
    cycle_figs = make_gt_roma_cycle_error_figs(roma_pred, data, n_pairs=n_pairs)
    matches_cycle_figs = make_gt_roma_matches_cycle_error_figs(
        roma_pred, data, n_pairs=n_pairs
    )
    matches_cycle_intersection_figs = (
        make_gt_roma_matches_cycle_error_intersection_figs(
            roma_pred, data, n_pairs=n_pairs
        )
    )

    for idx, name in enumerate(names):
        pair_dir = base_dir / "roma_gt_debug" / name
        _save_named_figure(raw_figs[idx], pair_dir / "raw_images.png")
        _save_named_figure(pure_warp_figs[idx], pair_dir / "roma_pure_warp.png")
        _save_named_figure(
            directional_figs["1to0"][idx], pair_dir / "roma_warp_1to0.png"
        )
        _save_named_figure(
            directional_figs["0to1"][idx], pair_dir / "roma_warp_0to1.png"
        )
        _save_named_figure(
            certainty_figs[idx], pair_dir / "roma_certainty_heatmap.png"
        )
        _save_named_figure(matches_figs[idx], pair_dir / "roma_matches_certainty.png")
        _save_named_figure(
            matches_intersection_figs[idx],
            pair_dir / "roma_matches_certainty_intersection.png",
        )
        if idx < len(matches_cycle_figs):
            _save_named_figure(
                matches_cycle_figs[idx], pair_dir / "roma_matches_cycle_error.png"
            )
        if idx < len(matches_cycle_intersection_figs):
            _save_named_figure(
                matches_cycle_intersection_figs[idx],
                pair_dir / "roma_matches_cycle_error_intersection.png",
            )
        if idx < len(cycle_figs):
            _save_named_figure(cycle_figs[idx], pair_dir / "roma_cycle_error.png")
        report = _build_roma_pair_report(idx, name, roma_pred, gt, data, conf)
        (pair_dir / "report.txt").write_text(report)


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
            "cycle_error_threshold": None,
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
                base_dir = Path(settings.TRAINING_PATH) / getattr(
                    self.conf, "experiment_name", "debug"
                )
                names = _prepare_fig_names(
                    data.get("names", data.get("idx", "pair")),
                    num_figs=data["keypoints0"].shape[0],
                )
                figs = make_gt_pos_neg_ign_figs(
                    gt,
                    data,
                    n_pairs=data["keypoints0"].shape[0],
                    pos_th=self.conf.th_positive,
                    neg_th=self.conf.th_negative,
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
                _save_roma_phase1_debug_outputs(roma_pred, gt, data, self.conf)
        return gt

    def loss(self, pred, data):
        raise NotImplementedError
