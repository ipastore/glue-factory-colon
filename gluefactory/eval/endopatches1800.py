from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from .hpatches import HPatchesPipeline
from .io import get_eval_parser, parse_eval_args
from ..settings import EVAL_PATH


class Endopatches1800Pipeline(HPatchesPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "homographies",
            "data_dir": "all_SLAM_ENE26",
            "num_workers": 4,
            "train_size": 150000,
            "val_size": 2000,
            "test_size": 1800,
            "test_seed": 0,
            "test_sequences": ["Seq_003", "Seq_016"],
            "test_homography_levels": [0.3, 0.5, 0.7],
            "test_photometric_levels": [0.25, 0.6, 0.95],
            "test_source_dir": "endopatches1800_source_images",
            "test_image_list": None,
            "save_dataset": False,
            "read_dataset_from_disk": False,
            "saved_dataset_dir": "endopatches1800",
            "reseed": True,
            "right_only": False,
            "left_view_difficulty": 0.0,
            "right_view_difficulty": None,
            "crop_vignette": True,
            "vignette_crop_coords": [81, 663, 55, 484],
            "photometric": {
                "name": "lg",
            },
            "homography": {
                "difficulty": 0.7,
                "max_angle": 45,
                "patch_shape": [580, 426],
            },
        },
        "model": {
            "ground_truth": {
                "name": None,
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": -1,
        },
    }

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = self._get_dataset(data_conf)
        return dataset.get_data_loader("test")

    @staticmethod
    def _get_dataset(data_conf):
        from ..datasets import get_dataset

        return get_dataset(data_conf["name"])(data_conf)


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(Endopatches1800Pipeline.default_conf)

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

    pipeline = Endopatches1800Pipeline(conf)
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
