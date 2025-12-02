import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera, Pose
from ..settings import DATA_PATH
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_heatmaps, plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics
from ..utils.misc import pad_to_length


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
        "data_dir": "Endomapper_CUDASIFT_NOV25/",
        "npz_subpath": "processed_npz/",
        # Training
        "train_split": "train_seqs.txt",
        "train_num_per_seq": 500,
        # Validation
        "val_split": "val_seqs.txt",
        "val_num_per_seq": None,
        "val_pairs": None,
        # Test
        "test_split": "test_seqs.txt",
        "test_num_per_seq": None,
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
        # CudaSift features
        "max_num_features": 2048
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
            seqs_maps_path = seq_lists_path / split_conf
            seqs_maps = seqs_maps_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            seqs_maps = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        seqs_maps = sorted(set(seqs_maps))

        self.seq = {}
        self.map_id = {}
        self.image_names = {}
        self.poses = {}
        self.intrinsics = {}
        self.valid = {}
        self.dist_coeffs = {}
        self.point3D_ids={}
        self.point3D_coords={}
        self.overlap_matrix = {}

        # load data
        self.seqs_maps = []
        for seq_map in seqs_maps:
            path = self.conf.data_dir / self.conf.npz_subpath / (seq_map + ".npz")
            try:
                data_npz = np.load(str(path), allow_pickle=True)
            except Exception:
                logger.warning(
                    "Cannot load seq_map data for seq_map %s at %s.", seq_map, path
                )
                continue
            self.image_names[seq_map] = data_npz["image_names"]
            self.poses[seq_map] = data_npz["poses"]
            self.intrinsics[seq_map] = data_npz["intrinsics"]
            self.map_id[seq_map] = data_npz["map_id"]
            self.seq[seq_map] = data_npz["seq"]
            self.point3D_ids[seq_map] = data_npz["point3D_ids_all"]
            self.point3D_coords[seq_map] = data_npz["point3D_coords_all"]
            self.dist_coeffs[seq_map] = data_npz["dist_coeffs"]
            self.overlap_matrix[seq_map] = data_npz["overlap_matrix"]

            self.seqs_maps.append(seq_map)

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []
        split = self.split
        num_per_seq = self.conf[self.split + "_num_per_seq"]
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
                assert im1.split("/")[0] == seq
                assert im0 in self.image_names[seq]
                assert im1 in self.image_names[seq]
                idx0 = np.where(self.image_names[seq] == im0)[0][0]
                idx1 = np.where(self.image_names[seq] == im1)[0][0]
                self.items.append((seq, idx0, idx1, 1.0))
        #Not tested this if statement
        elif self.conf.views == 1:
            for seq in self.seqs_maps:
                if seq not in self.image_names:
                    continue
                valid = (self.image_names[seq] != None) | (  # noqa: E711
                    self.depths[seq] != None  # noqa: E711
                )
                ids = np.where(valid)[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                ids = [(seq, i) for i in ids]
                self.items.extend(ids)
        else:
            for seq_map in self.seqs_maps:

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
                    pairs = (self.overlap_matrix > self.conf.min_overlap) & (
                        self.overlap_matrix <= self.conf.max_overlap
                    )
                    pairs = np.stack(np.where(pairs), -1)

                pairs = [(seq_map, self.image_names[i], self.image_names[j], self.overlap_matrix[i, j]) for i, j in pairs]
                if num_neg is not None:
                    neg_pairs = np.stack(np.where(self.overlap_matrix <= 0.0), -1)
                    neg_pairs = sample_n(neg_pairs, num_neg, seed)
                    pairs += [(seq, self.image_names[i], self.image_names[j], self.overlap_matrix[i, j]) for i, j in neg_pairs]
                self.items.extend(pairs)
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, seq_map, idx):
        
        path = self.conf.data_dir / self.conf.npz_subpath / (seq_map + ".npz")
        data_npz = np.load(str(path), allow_pickle=True)

        # read pose data
        K = self.intrinsics[seq_map][idx].astype(np.float32, copy=False)
        T = self.poses[seq_map][idx].astype(np.float32, copy=False)
        name = self.image_names[seq_map][idx]
        sparse_depth = data_npz["depths_per_image"][idx]
        keypoints = data_npz["keypoints_per_image"][idx]  
        descriptors = data_npz["descriptors_per_image"][idx]  
        scales = data_npz["scales_per_image"][idx]  
        orientations = data_npz["orientations_per_image"][idx]  
        keypoint_scores = np.abs(data_npz["scores_per_image"][idx]) * scales
        point3D_ids = data_npz["point3D_ids_per_image"][idx]
        valid_depth_mask = data_npz["valid_depth_mask_per_image"][idx]
        valid_3D_mask = data_npz["valid_3d_mask_per_image"][idx]

        # Assert all feature arrays have the same length in first dimension
        assert len({sparse_depth.shape[0], keypoints.shape[0], descriptors.shape[0], 
                    scales.shape[0], orientations.shape[0], keypoint_scores.shape[0], 
                    point3D_ids.shape[0], valid_depth_mask.shape[0], valid_3D_mask.shape[0]}) == 1, \
            "Feature arrays have mismatched lengths in first dimension"

        pred = {
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

        # # Not tested yet: add random rotations 
        # do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        # if do_rotate:
        #     p = self.conf.p_rotate
        #     k = 0
        #     if np.random.rand() < p:
        #         k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
        #         img = torch.rot90(img, k=-k, dims=[1, 2])
        #         depth = torch.rot90(depth, k=-k, dims=[1, 2]).clone()
        #         K = rotate_intrinsics(K, img.shape, k + 2)
        #         T = rotate_pose_inplane(T, k + 2)
        
        # Truncate features based on scores
        max_num_features = self.conf.get("max_num_features", None)
        if len(keypoints) > max_num_features:
            indices = torch.topk(keypoint_scores, max_num_features).indices
        pred = {k: v[indices] for k, v in pred.items()}

        # Padding for cnsistent batching
        # Pad with zeros (creates invalid keypoints at [0, 0])
        pred["keypoints"] = pad_to_length(
            pred["keypoints"], max_num_features, -2, mode="zeros"
        )
        pred["descriptors"] = pad_to_length(
            pred["descriptors"], max_num_features, -2, mode="zeros"
        )
        pred["scales"] = pad_to_length(
            pred["scales"], max_num_features, -1, mode="zeros"
        )
        pred["oris"] = pad_to_length(
            pred["oris"], max_num_features, -1, mode="zeros"
        )
        pred["keypoint_scores"] = pad_to_length(
            pred["keypoint_scores"], max_num_features, -1, mode="zeros"
        )
        # Pad depth with -1.0 (MISSING_DEPTH_VALUE)
        pred["depth"] = pad_to_length(
            pred["depth"], max_num_features, -1, mode="constant", constant=-1.0
        )
        # Pad point3D_ids with -1 (invalid ID)
        pred["point3D_ids"] = pad_to_length(
            pred["point3D_ids"], max_num_features, -1, mode="constant", constant=-1
        )
        # Pad masks with False (invalid)
        pred["valid_depth_mask"] = pad_to_length(
            pred["valid_depth_mask"], max_num_features, -1, mode="constant", constant=False
        )
        pred["valid_3D_mask"] = pad_to_length(
            pred["valid_3D_mask"], max_num_features, -1, mode="constant", constant=False
        )
        

        data = {
            "name": name,
            "seq_map": seq_map,
            "T_w2cam": Pose.from_4x4mat(T),
            "camera": Camera.from_calibration_matrix(K).float(),
            **pred,
        }
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
                seq_map, idx0, idx1, overlap = idx
            else:
                seq_map, idx0, idx1, overlap = self.items[idx]
            data0 = self._read_view(seq_map, idx0)
            data1 = self._read_view(seq_map, idx1)
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["overlap_0to1"] = overlap
            data["name"] = f"{seq_map}/{data0['name']}_{data1['name']}"
        else:
            assert self.conf.views == 1
            seq_map, idx0 = self.items[idx]
            data = self._read_view(seq_map, idx0)
        data["seq_map"] = seq_map
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
        images, depths = [], []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )
            depths.append(
                [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
            )

    axes = plot_image_grid(images, dpi=args.dpi)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
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
