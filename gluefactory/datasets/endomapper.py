import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera, Pose
from ..settings import DATA_PATH
from ..utils.image import load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset
from .utils import scale_intrinsics
from ..models.utils.misc import pad_to_length


logger = logging.getLogger(__name__)
seq_lists_path = Path(__file__).parent / "endomapper_seq_lists"


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class Endomapper(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "Endomapper_CUDASIFT",
        "npz_subpath": "processed_npz",
        # Training
        "train_split": "train_seqs_maps.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "val_seqs_maps.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
        # Test
        "test_split": "test_seqs_maps.txt",
        "test_num_per_scene": None,
        "test_pairs": None,
        # data sampling
        "views": 2,     # Triplet Dataset not implemented
        "min_overlap": 0.3,  # only with D2-Net format
        "max_overlap": 1.0,  # only with D2-Net format
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,  #Not implemented only with views==3
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°, Not used, nor implemented
        "reseed": False,
        "seed": 0,
        "read_image": False,
        "grayscale": False,
        # CudaSift features
        "max_num_features": 2048,
        "min_images_per_map": 10,
        "min_3D_points_per_map": 50
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
            path_not_found = DATA_PATH / conf.data_dir
            raise FileExistsError(f"Endomapper Dataset not found in {path_not_found} ")

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            raise NotImplementedError("Triplet Dataset not implemented for Endomapper")
        else:
            return _PairDataset(self.conf, split)

class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):
            seqs_path = seq_lists_path / split_conf
            seqs = seqs_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            seqs = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        seqs = sorted(set(seqs))

        seqs_maps = []
        npz_dir = self.root / self.conf.npz_subpath
        for seq in seqs:
            matching_files = sorted(npz_dir.glob(f"{seq}_map*.npz"))
            if not matching_files:
                logger.warning(f"No maps found for sequence {seq} in {npz_dir}")
                continue
            for npz_file in matching_files:
                seq_map = npz_file.stem
                seqs_maps.append(seq_map)
        
        logger.info(f"Found {len(seqs_maps)} maps from {len(seqs)} sequences for {split} split")


        self.seq: Dict[str, str] = {}
        self.map_id: Dict[str, str] = {}
        self.image_names: Dict[str, np.ndarray] = {}
        self.image_sizes: Dict[str, np.ndarray] = {}
        self.camera_ids: Dict[str, np.ndarray] = {}
        self.poses: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.valid: Dict[str, np.ndarray] = {}
        self.dist_coeffs: Dict[str, np.ndarray] = {}
        self.point3D_ids_all: Dict[str, np.ndarray] = {}
        self.point3D_coords_all: Dict[str, np.ndarray] = {}
        self.overlap_matrix: Dict[str, np.ndarray] = {}
        self.keypoints: Dict[str, np.ndarray] = {}
        self.descriptors: Dict[str, np.ndarray] = {}
        self.depths: Dict[str, np.ndarray] = {}
        self.scales: Dict[str, np.ndarray] = {}
        self.orientations: Dict[str, np.ndarray] = {}
        self.scores: Dict[str, np.ndarray] = {}
        self.point3D_ids_per_image: Dict[str, np.ndarray] = {}
        self.valid_depth_mask: Dict[str, np.ndarray] = {}
        self.valid_3d_mask: Dict[str, np.ndarray] = {}

        self.seqs_maps = []
        for seq_map in seqs_maps:
            path = self.root / self.conf.npz_subpath / f"{seq_map}.npz"
            try:
                data_npz = np.load(str(path), allow_pickle=True)
            except Exception:
                logger.warning(
                    "Cannot load seq_map data for %s at %s", seq_map, path
                )
                continue

            len_images = data_npz["image_names"].shape[0]
            len_3D_points = data_npz["point3D_ids"].shape[0]
            if len_images < self.conf.min_images_per_map or len_3D_points < self.conf.min_3D_points_per_map:
                continue

            self.image_names[seq_map] = data_npz["image_names"]
            self.image_sizes[seq_map] = data_npz["image_sizes"]
            self.camera_ids[seq_map] = data_npz["camera_ids"]
            self.poses[seq_map] = data_npz["poses"]
            self.intrinsics[seq_map] = data_npz["intrinsics"]

            self.map_id[seq_map] = str(np.asarray(data_npz["map_id"]).item())
            self.seq[seq_map] = str(np.asarray(data_npz["seq"]).item())
            self.point3D_ids_all[seq_map] = data_npz["point3D_ids"]
            self.point3D_coords_all[seq_map] = data_npz["point3D_coords"]
            self.dist_coeffs[seq_map] = data_npz["distortion_coeffs"]
            self.overlap_matrix[seq_map] = data_npz["overlap_matrix"].astype(
                np.float32, copy=False
            )

            self.seqs_maps.append(seq_map)

        if not self.seqs_maps:
            raise ValueError("No Endomapper sequences loaded for split.")

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []
        split = self.split
        num_per_seq = self.conf[self.split + "_num_per_scene"]
        if isinstance(num_per_seq, Iterable):
            num_pos, num_neg = num_per_seq
        else:
            num_pos = num_per_seq
            num_neg = None
        # Not tested this if statement
        if split != "train" and self.conf[split + "_pairs"] is not None:
            # Fixed validation or test pairs
            assert num_pos is None
            assert num_neg is None
            assert self.conf.views == 2
            pairs_path = seq_lists_path / self.conf[split + "_pairs"]
            for line in pairs_path.read_text().rstrip("\n").split("\n"):
                im0, im1 = line.split(" ")
                seq = im0.split("/")[0]
                if seq not in self.image_names or im1.split("/")[0] != seq:
                    continue
                im0_name = im0.split("/")[-1]
                im1_name = im1.split("/")[-1]
                if im0_name not in self.image_names[seq] or im1_name not in self.image_names[seq]:
                    continue
                idx0 = np.where(self.image_names[seq] == im0_name)[0][0]
                idx1 = np.where(self.image_names[seq] == im1_name)[0][0]
                self.items.append((seq, idx0, idx1, 1.0))
        #Not tested this if statement
        elif self.conf.views == 1:
            for seq in self.seqs_maps:
                if seq not in self.image_names:
                    continue
                valid = self.valid.get(seq, None)
                if valid is None:
                    continue
                ids = np.where(valid)[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                ids = [(seq, i) for i in ids]
                self.items.extend(ids)
        else:
            for seq_map in self.seqs_maps:
                mat = self.overlap_matrix[seq_map]
                if num_pos is not None:
                    # Sample a subset of pairs, binned by overlap.
                    num_bins = self.conf.num_overlap_bins
                    assert num_bins > 0
                    bin_width = (
                        self.conf.max_overlap - self.conf.min_overlap
                    ) / num_bins
                    num_per_bin = num_pos // num_bins
                    pairs_all = []
                    for k in range(num_bins):
                        bin_min = self.conf.min_overlap + k * bin_width
                        bin_max = bin_min + bin_width
                        pairs_bin = (mat > bin_min) & (mat <= bin_max)
                        pairs_bin = np.stack(np.where(pairs_bin), -1)
                        pairs_all.append(pairs_bin)
                    # Skip bins with too few samples
                    has_enough_samples = [len(p) >= num_per_bin * 2 for p in pairs_all]
                    num_per_bin_2 = num_pos // max(1, sum(has_enough_samples))
                    pairs = []
                    for pairs_bin, keep in zip(pairs_all, has_enough_samples):
                        if keep:
                            pairs.append(sample_n(pairs_bin, num_per_bin_2, seed))
                    pairs = np.concatenate(pairs, 0)
                else:
                    pairs = (mat > self.conf.min_overlap) & (
                        mat <= self.conf.max_overlap
                    )
                    pairs = np.stack(np.where(pairs), -1)
                
                image_names = self.image_names[seq_map]
                pairs = [(seq_map, image_names[i], image_names[j], mat[i, j]) for i, j in pairs]
                if num_neg is not None:
                    neg_pairs = np.stack(np.where(self.overlap_matrix <= 0.0), -1)
                    neg_pairs = sample_n(neg_pairs, num_neg, seed)
                    pairs += [(seq, image_names[i], image_names[j], mat[i, j]) for i, j in neg_pairs]
                self.items.extend(pairs)
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, seq_map, image_name):
        
        image_names = self.image_names[seq_map]
        idx = np.where(image_names == image_name)[0][0]
        path = self.root / self.conf.npz_subpath / f"{seq_map}.npz"
        data_npz = np.load(str(path), allow_pickle=True)

        # read pose data
        K = self.intrinsics[seq_map][idx].astype(np.float32, copy=False)
        T = self.poses[seq_map][idx].astype(np.float32, copy=False)
        name = str(self.image_names[seq_map][idx])
        image_size = torch.tensor(self.image_sizes[seq_map][idx]).float()
        image = None
        if self.conf.read_image:
            name = f"Keyframe_{name}.png"
            image_path = (
                self.root
                / self.seq[seq_map]
                / "output"
                / "3D_maps"
                / self.map_id[seq_map]
                / "keyframes"
                / name
            )
            def _dummy_image():
                c = 1 if self.conf.grayscale else 3
                w, h = int(image_size[0].item()), int(image_size[1].item())
                return torch.zeros((c, h, w), dtype=torch.float32)
            if not image_path.exists():
                logger.warning("Image not found at %s, using dummy.", image_path)
                image = _dummy_image()
            else:
                try:
                    image = load_image(image_path, grayscale=self.conf.grayscale)
                except Exception:
                    logger.warning("Failed to load image at %s, using dummy.", image_path)
                    image = _dummy_image()
        sparse_depth = torch.from_numpy(data_npz["depths_per_image"][idx]).float()
        keypoints = torch.from_numpy(data_npz["keypoints_per_image"][idx]).float()  
        descriptors = torch.from_numpy(data_npz["descriptors_per_image"][idx]).float()  
        scales = torch.from_numpy(data_npz["scales_per_image"][idx]).float()  
        orientations = torch.from_numpy(data_npz["orientations_per_image"][idx]).float()
        orientations = orientations * (torch.pi / 180.0)
        point3D_ids = torch.from_numpy(data_npz["point3D_ids_per_image"][idx]).float()
        valid_depth_mask = torch.from_numpy(data_npz["valid_depth_mask_per_image"][idx]).bool()
        valid_3D_mask = torch.from_numpy(data_npz["valid_3d_mask_per_image"][idx]).bool()
        keypoint_scores = torch.from_numpy(np.abs(data_npz["scores_per_image"][idx])).float() * scales

        # Assert all feature arrays have the same length in first dimension
        lengths = {
            keypoints.shape[0],
            descriptors.shape[0],
            sparse_depth.shape[0],
            scales.shape[0],
            orientations.shape[0],
            keypoint_scores.shape[0],
            point3D_ids.shape[0],
            valid_depth_mask.shape[0],
            valid_3D_mask.shape[0],
        }
        assert len(lengths) == 1, "Feature arrays have mismatched lengths."

        cache = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "sparse_depth": sparse_depth,
            "scales": scales,
            "oris": orientations,
            "keypoint_scores": keypoint_scores,
            "point3D_ids": point3D_ids,
            "valid_depth_mask": valid_depth_mask,
            "valid_3D_mask": valid_3D_mask,
        }

        
        # Truncate features based on scores
        max_num_features = self.conf.get("max_num_features", None)
        if max_num_features is None:
            raise ValueError("max_num_features must be not None")
        if len(keypoints) > max_num_features:
            indices = torch.topk(keypoint_scores, max_num_features).indices
        elif len(keypoints) == 0:
            indices = torch.tensor([], dtype=torch.long)
        else:
            indices = torch.arange(len(keypoints), dtype=torch.long)
        cache = {k: v[indices] for k, v in cache.items()}

        # Padding for cnsistent batching
        # Pad with zeros (creates invalid keypoints at [0, 0])
        cache["keypoints"] = pad_to_length(
            cache["keypoints"], 
            max_num_features, 
            -2,
            mode="random_c",
            bounds=(0, min(image_size))
        )
        cache["descriptors"] = pad_to_length(
            cache["descriptors"], max_num_features, -2, mode="random"

        )
        cache["scales"] = pad_to_length(
            cache["scales"], max_num_features, -1, mode="zeros"
        )
        cache["oris"] = pad_to_length(
            cache["oris"], max_num_features, -1, mode="zeros"
        )
        cache["keypoint_scores"] = pad_to_length(
            cache["keypoint_scores"], max_num_features, -1, mode="zeros"
        )
        # Pad depth with -1.0 (MISSING_DEPTH_VALUE)
        cache["sparse_depth"] = pad_to_length(
            cache["sparse_depth"], max_num_features, -1, mode="minus_one"
        )
        # Pad point3D_ids with -1 (invalid ID)
        cache["point3D_ids"] = pad_to_length(
            cache["point3D_ids"], max_num_features, -1, mode="minus_one"
        )
        # Pad masks with False (invalid)
        cache["valid_depth_mask"] = pad_to_length(
            cache["valid_depth_mask"], max_num_features, -1, mode=False
        )
        cache["valid_3D_mask"] = pad_to_length(
            cache["valid_3D_mask"], max_num_features, -1, mode=False
        )
        

        data = {
            "name": name,
            "seq_map": seq_map,
            "T_w2cam": Pose.from_4x4mat(T),
            "camera": Camera.from_calibration_matrix(K).float(),
            "image_size": image_size, #WxH
        }
        if image is not None:
            data["image"] = image
        data = {"cache": cache, **data}
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        if self.conf.views == 2:
            if isinstance(idx, list):
                seq_map, img_name0, img_name1, overlap = idx
            else:
                seq_map, img_name0, img_name1, overlap = self.items[idx]
            data0 = self._read_view(seq_map, img_name0)
            data1 = self._read_view(seq_map, img_name1)
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["overlap_0to1"] = overlap
            data["names"] = f"{seq_map}/{data0['name']}_{data1['name']}"
        else:
            assert self.conf.views == 1
            seq_map, idx0 = self.items[idx]
            data = self._read_view(seq_map, idx0)
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = Endomapper(conf)
    loader = dataset.get_data_loader(args.split)
    logger.info("The dataset has elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )

    axes = plot_image_grid(images, dpi=args.dpi)
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
