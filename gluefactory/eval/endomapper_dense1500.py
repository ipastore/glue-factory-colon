import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_matches_depth,
    eval_poses,
    eval_relative_pose_robust,
)



class EndomapperDense1500Pipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "posed_images",
            "root": "",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/depths",
            "views": "{scene}/views.txt",
            "view_groups": "{scene}/pairs.txt",
            "depth_format": "npz",
            "crop_endomapper_dense": True,
            "depth_scale_scene_info_dir": "endomapper_dense/scene_info",
            "read_specular_mask": True,
            "specular_scene_info_dir": "endomapper_dense/scene_info",
            "scene_list": ["endomapper_dense1500"],
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0, # -1 runs a bunch of thresholds and selects the best
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        min_matches_for_pose = 5
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for _, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            num_kpts0 = int(pred["keypoints0"].shape[0])
            num_kpts1 = int(pred["keypoints1"].shape[0])
            num_matches = int((pred["matches0"] > -1).sum().item())

            results_i = {}
            results_i["num_matches"] = num_matches
            results_i["num_keypoints"] = 0.5 * (num_kpts0 + num_kpts1)
            if "depth" in data["view0"].keys():
                results_i.update(eval_matches_depth(data, pred))

            for th in test_thresholds:
                if num_matches < min_matches_for_pose:
                    pose_results_i = {
                        "rel_pose_error": float("nan"),
                        "ransac_inl": float("nan"),
                        "ransac_inl%": float("nan"),
                    }
                else:
                    pose_results_i = eval_relative_pose_robust(
                        data,
                        pred,
                        {"estimator": conf.estimator, "ransac_th": th},
                    )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]
            for k, v in results_i.items():
                results[k].append(v)

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.nanmean(arr), 3) if np.any(np.isfinite(arr)) else np.nan

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        for k, v in best_pose_results.items():
            summaries[k] = v

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }
        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(EndomapperDense1500Pipeline.default_conf)
    
    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = EndomapperDense1500Pipeline(conf)
    s, f, _ = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
