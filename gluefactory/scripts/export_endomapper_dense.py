import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..geometry.depth import sample_depth
from ..models import get_model
from ..settings import DATA_PATH
from ..utils.export_predictions import export_predictions
from .export_megadepth import configs, resize

SEQ_LISTS_PATH = Path(__file__).resolve().parents[1] / "datasets" / "endomapper_dense_seq_lists"

n_kpts = 2048
configs = {
    "sp": {
        "name": f"SP-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "gluefactory_nonfree.superpoint",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "sp_open": {
        "name": f"SP-open-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "extractors.superpoint_open",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "cv2-sift": {
        "name": f"opencv-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": 4096,
            "backend": "opencv",
        },
    },
    "pycolmap-sift": {
        "name": f"pycolmap-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap",
        },
    },
    "pycolmap-sift-gpu": {
        "name": f"pycolmap_SIFTGPU-nms3-fixed-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap_cuda",
            "nms_radius": 3,
            "force_num_keypoints": False,
            "detection_threshold": 0.0000667,
            "rootsift": True,
            "first_octave": -1,
            "num_octaves": 4,
            "init_blur": 1.0,
            "extractor_channel": "grayscale",
        },
    },
    "py-cudasift": {
        # "name": f"py-cudasift-k{n_kpts}",
        "name": f"py-cudasift-k{n_kpts}_no_scale",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "backend": "py_cudasift",
            "max_num_keypoints": n_kpts,
            "force_num_keypoints": False,
            "nms_radius": 3,
            "detection_threshold": 0.0000667,
            "rootsift": True,
            "first_octave": -1,
            "num_octaves": 4,
            "init_blur": 1.0,
            "extractor_channel": "grayscale",
            "filter_kpts_with_wrapper": False ,          # Only for py_cudasift. Truncate max kpts as Cudasift
            "filter_with_scale_weighting": False ,     # for all that has scores. Multiply scores by scales.
            "filter_with_lowest_scale": False ,     # Default is false. Only for those who dont have scores, scale as filter proxy.  
            "random_topk": False # if True, pick random topk even when scores are available 
        },
    },
    "keynet-affnet-hardnet": {
        "name": f"KeyNetAffNetHardNet-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.keynet_affnet_hardnet",
            "max_num_keypoints": n_kpts,
        },
    },
    "disk": {
        "name": f"DISK-k{n_kpts}-nms5",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.disk_kornia",
            "max_num_keypoints": n_kpts,
        },
    },
    "aliked": {
        "name": f"ALIKED-k{n_kpts}-n16",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.aliked",
            "max_num_keypoints": n_kpts,
        },
    },
}

def _read_seq_maps(args) -> list[str]:
    if args.seq_maps:
        return list(args.seq_maps)
    split_path = Path(args.split_file)
    split_path = SEQ_LISTS_PATH / split_path
    lines = [l.strip() for l in split_path.read_text().splitlines() if l.strip()]
    return lines


def _load_scale_by_image(scene_info_path: Path) -> dict[str, float]:
    with np.load(str(scene_info_path), allow_pickle=True) as scene_info:
        image_names = [str(x) for x in scene_info["image_names"].tolist()]
        scales = scene_info["depth_scale_per_image"].astype(np.float32, copy=False)
    return {name: float(scales[i]) for i, name in enumerate(image_names)}


# def _resolve_scale(scale_by_image: dict[str, float], image_name: str) -> float:
#     if image_name in scale_by_image:
#         return scale_by_image[image_name]
#     image_base = Path(image_name).name
#     if image_base in scale_by_image:
#         return scale_by_image[image_base]
#     return 1.0


def _get_kp_depth_with_scale(scale_by_image: dict[str, float]):
    def _callback(pred, data):
        d, valid = sample_depth(pred["keypoints"], data["depth"])
        image_name = data["name"][0]
        scale = scale_by_image[image_name]
        return {
            "depth_keypoints": d * float(scale),
            "valid_depth_keypoints": valid,
        }

    return _callback


def run_export(feature_file: Path, seq_map: str, args):
    conf = {
        "data": {
            "name": "endomapper_dense",
            "views": 1,
            "grayscale": configs[args.method]["gray"],
            "batch_size": 1,
            "num_workers": args.num_workers,
            "read_depth": True,
            "read_image": True,
            "train_split": [seq_map],
            "train_num_per_scene": None,
        },
        "split": "train",
        "model": configs[args.method]["conf"],
    }
    conf = OmegaConf.create(conf)

    keys = configs[args.method]["keys"]
    keys = keys + ["depth_keypoints", "valid_depth_keypoints"]

    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(conf.split or "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(conf.model.name)(conf.model).eval().to(device)

    scene_info_path = DATA_PATH / "endomapper_dense" / "scene_info" / f"{seq_map}.npz"

    scale_by_image = _load_scale_by_image(scene_info_path)
    callback_fn = _get_kp_depth_with_scale(scale_by_image)

    export_predictions(
        loader, model, feature_file, as_half=True, keys=keys, callback_fn=callback_fn
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="py-cudasift")
    parser.add_argument("--split_file", type=str, default="train_seqs.txt")
    parser.add_argument("--seq_maps", type=str, nargs="*", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    export_name = configs[args.method]["name"]
    export_root = Path(DATA_PATH,"exports","endomapper-dense-" + export_name)
    export_root.mkdir(parents=True, exist_ok=True)

    seq_maps = _read_seq_maps(args)
    for i, seq_map in enumerate(seq_maps):
        print(f"{i} / {len(seq_maps)}", {seq_map})
        feature_file = export_root / f"{seq_map}.h5"
        if feature_file.exists() and not args.overwrite:
            continue
        logging.info("Export local features for seq_map %s", seq_map)
        run_export(feature_file, seq_map, args)
