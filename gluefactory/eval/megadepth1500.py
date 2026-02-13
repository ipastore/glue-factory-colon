import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_matches_depth,
    eval_matches_epipolar,
    eval_poses,
    eval_relative_pose_robust,
)

logger = logging.getLogger(__name__)


class MegaDepth1500Pipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "posed_images",
            "root": "",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/depths",
            "views": "{scene}/views.txt",
            "view_groups": "{scene}/pairs.txt",
            "depth_format": "h5",
            "scene_list": ["megadepth1500"],
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        # "keypoint_scores0",
        # "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        if not (DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()

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
        counts = defaultdict(int)
        min_matches_for_pose = 5
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            counts["num_total_pairs"] += 1
            pred = cache_loader(data)
            num_kpts0 = int(pred["keypoints0"].shape[0])
            num_kpts1 = int(pred["keypoints1"].shape[0])
            num_matches = int((pred["matches0"] > -1).sum().item())
            if num_kpts0 == 0 or num_kpts1 == 0:
                counts["num_pairs_no_keypoints"] += 1
            if num_matches > 0:
                counts["valid_pairs_epi"] += 1
            # add custom evaluations here
            results_i = eval_matches_epipolar(data, pred)
            if "depth" in data["view0"].keys():
                results_i.update(eval_matches_depth(data, pred))
                if np.isfinite(results_i["reproj_prec@1px"]):
                    counts["valid_pairs_reproj"] += 1
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
            if num_matches >= min_matches_for_pose:
                counts["valid_pairs_pose"] += 1

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
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
        combined_summaries = {**summaries, **best_pose_results}
        summaries = {"num_total_pairs": int(counts["num_total_pairs"])}
        inserted_epi = False
        inserted_reproj = False
        inserted_pose = False
        for k, v in combined_summaries.items():
            summaries[k] = v
            if k == "mepi_prec@1e-3":
                summaries["valid_pairs_epi"] = int(counts["valid_pairs_epi"])
                inserted_epi = True
            if k == "mreproj_prec@5px":
                summaries["valid_pairs_reproj"] = int(counts["valid_pairs_reproj"])
                inserted_reproj = True
            if k == "mrel_pose_error":
                summaries["valid_pairs_pose"] = int(counts["valid_pairs_pose"])
                inserted_pose = True
        if not inserted_epi:
            summaries["valid_pairs_epi"] = int(counts["valid_pairs_epi"])
        if not inserted_reproj:
            summaries["valid_pairs_reproj"] = int(counts["valid_pairs_reproj"])
        if not inserted_pose:
            summaries["valid_pairs_pose"] = int(counts["valid_pairs_pose"])

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
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MegaDepth1500Pipeline.default_conf)

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

    pipeline = MegaDepth1500Pipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
