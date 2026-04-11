import argparse
import logging
import os
import signal
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics



logger = logging.getLogger(__name__)
seq_lists_path = Path(__file__).parent / "endomapper_roma_seq_lists"


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class EndomapperRoma(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "slam-results_long_sequences_ENE26",
        "npz_subpath": "processed_npz",
        # Training
        "train_split": "train_seqs.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "val_seqs.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
        # Overfit (val-like) split
        "overfit_split": None,
        "overfit_num_per_scene": None,
        "overfit_pairs": None,
        # Test
        "test_split": "test_seqs.txt",
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
        "read_image": True,
        "grayscale": False,
        "min_images_per_map": 10,
        # "min_3D_points_per_map": 50
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
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
            direct_npz = npz_dir / f"{seq}.npz"
            if direct_npz.exists():
                seqs_maps.append(direct_npz.stem)
                continue
            matching_files = sorted(npz_dir.glob(f"{seq}_map*.npz"))
            if not matching_files:
                logger.warning(f"No maps found for sequence {seq} in {npz_dir}")
                continue
            for npz_file in matching_files:
                seq_map = npz_file.stem
                seqs_maps.append(seq_map)
        
        logger.info(f"Found {len(seqs_maps)} maps from {len(seqs)} sequences for {split} split")

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.seq: Dict[str, str] = {}
        self.map_id: Dict[str, str] = {}
        self.image_names: Dict[str, np.ndarray] = {}
        self.image_sizes: Dict[str, np.ndarray] = {}
        self.camera_ids: Dict[str, np.ndarray] = {}
        self.poses: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.valid: Dict[str, np.ndarray] = {}
        # self.point3D_ids_all: Dict[str, np.ndarray] = {}
        # self.point3D_coords_all: Dict[str, np.ndarray] = {}
        self.overlap_matrix: Dict[str, np.ndarray] = {}
        # self.keypoints: Dict[str, np.ndarray] = {}
        # self.descriptors: Dict[str, np.ndarray] = {}
        # self.depths: Dict[str, np.ndarray] = {}
        # self.scales: Dict[str, np.ndarray] = {}
        # self.orientations: Dict[str, np.ndarray] = {}
        # self.scores: Dict[str, np.ndarray] = {}
        # self.point3D_ids_per_image: Dict[str, np.ndarray] = {}
        # self.valid_depth_mask: Dict[str, np.ndarray] = {}
        # self.valid_3d_mask: Dict[str, np.ndarray] = {}
        self.cameras = {}
        self.camera_indices: Dict[str, np.ndarray] = {}


        self.seqs_maps = []
        for seq_map in seqs_maps:
            path = self.root / self.conf.npz_subpath / f"{seq_map}.npz"
            try:
                with np.load(str(path), allow_pickle=True) as data_npz:
                    len_images = data_npz["image_names"].shape[0]
                    # len_3D_points = data_npz["point3D_ids"].shape[0]
                    if (
                        len_images < self.conf.min_images_per_map
                        # or len_3D_points < self.conf.min_3D_points_per_map
                    ):
                        continue

                    self.image_names[seq_map] = data_npz["image_names"]
                    self.image_sizes[seq_map] = data_npz["image_sizes"]
                    self.camera_ids[seq_map] = data_npz["camera_ids"]
                    self.poses[seq_map] = data_npz["poses"]
                    self.intrinsics[seq_map] = data_npz["intrinsics"]
                    self.map_id[seq_map] = str(np.asarray(data_npz["map_id"]).item())
                    self.seq[seq_map] = str(np.asarray(data_npz["seq"]).item())
                    # self.point3D_ids_all[seq_map] = data_npz["point3D_ids"]
                    # self.point3D_coords_all[seq_map] = data_npz["point3D_coords"]
                    self.overlap_matrix[seq_map] = data_npz["overlap_matrix"].astype(
                        np.float32, copy=False
                    )
                    self.cameras[seq_map] = data_npz["cameras"]
                    self.camera_indices[seq_map] = data_npz["camera_indices"]


            except Exception:
                logger.warning(
                    "Cannot load seq_map data for %s at %s", seq_map, path
                )
                continue

            self.valid[seq_map] = self._compute_valid(seq_map, len_images)
            self.seqs_maps.append(seq_map)

        if not self.seqs_maps:
            raise ValueError("No Endomapper sequences loaded for split.")

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0
    
    # Due to some error of extracting the video with fmpeg the last keyframe
    #  it´s not extracted, but it is in the npz, so we need to check if the keyframe exists,
    #  if not we mark as invalid, and skip it in the dataloader
    def _compute_valid(self, seq_map, len_images):
        valid = np.ones(len_images, dtype=bool)
        
        if self.conf.read_image:
            keyframes_dir = (
                self.root
                / self.seq[seq_map]
                / "output"
                / "3D_maps"
                / self.map_id[seq_map]
                / "keyframes"
            )
            image_valid = []
            for image_name in self.image_names[seq_map]:
                image_path = keyframes_dir / f"Keyframe_{str(image_name)}.png"
                if not image_path.exists():
                    image_valid.append(False)
                    continue
                try:
                    with PIL.Image.open(image_path) as image:
                        image.verify()
                except Exception:
                    image_valid.append(False)
                    continue
                image_valid.append(True)
            valid &= np.asarray(image_valid, dtype=bool)
        # if self.conf.read_depth:
        #     depth_exists = np.fromiter(
        #         ((self.root / str(path)).exists() for path in self.depths[seq_map]),
        #         dtype=bool,
        #         count=n,
        #     )
        #     valid &= depth_exists
        # if self.conf.read_specular_mask:
        #     specular_exists = np.fromiter(
        #         (
        #             (self.root / str(path)).exists()
        #             for path in self.specular_masks[seq_map]
        #         ),
        #         dtype=bool,
        #         count=n,
        #     )
        #     valid &= specular_exists
        return valid
    

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
                seq = line.split("/")[0]
                im0_name = str(line.split("/")[1].split("_")[1].strip(".png"))
                im1_name = str(line.split("/")[1].split("_")[-1].strip(".png"))
                if seq not in self.image_names:
                    continue
                if im0_name not in self.image_names[seq] or im1_name not in self.image_names[seq]:
                    continue
                idx0 = np.where(self.image_names[seq] == im0_name)[0]
                idx1 = np.where(self.image_names[seq] == im1_name)[0]
                if len(idx0) == 0 or len(idx1) == 0:
                    continue
                if not (self.valid[seq][idx0[0]] and self.valid[seq][idx1[0]]):
                    continue
                overlap = self.overlap_matrix[seq][idx0[0], idx1[0]]
                self.items.append((seq, im0_name, im1_name, overlap))
        elif self.conf.views == 1:
            for seq_map in self.seqs_maps:
                if seq_map not in self.image_names:
                    continue
                valid = self.valid.get(seq_map, None)
                if valid is None:
                    continue
                ids = np.where(valid)[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                
                image_names = self.image_names[seq_map]
                ids = [(seq_map, image_names[i]) for i in ids]
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
                    if not any(has_enough_samples):
                        logger.warning(
                            "Skipping %s: no bins with enough pairs for sampling.",
                            seq_map,
                        )
                        continue
                    if not all(has_enough_samples):
                        used_bins = [
                            str(i + 1)
                            for i, keep in enumerate(has_enough_samples)
                            if keep
                        ]
                        logger.warning(
                            "Sampling %s with bins %s.",
                            seq_map,
                            ",".join(used_bins),
                        )
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
    def _load_camera(self, seq_map, idx):
        cam_idx = int(np.asarray(self.camera_indices[seq_map][idx]).item())
        return Camera.from_npz(self.cameras[seq_map][cam_idx]).float()
    
    def _read_view(self, seq_map, image_name):
        
        image_names = self.image_names[seq_map]
        idx = int(np.where(image_names == image_name)[0][0])
        T = self.poses[seq_map][idx].astype(np.float32, copy=False)
        K = self.intrinsics[seq_map][idx].astype(np.float32, copy=True)
        camera = self._load_camera(seq_map, idx)

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

            
            image = load_image(image_path, self.conf.grayscale)
        else:
            size = PIL.Image.open(image_path).size[::-1]
            image = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()
        
        # add random rotations
        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = torch.rot90(img, k=-k, dims=[1, 2])
                K = rotate_intrinsics(K, img.shape, k + 2)
                T = rotate_pose_inplane(T, k + 2)


        data = {
            "name": name,
            "seq_map": seq_map,
            "T_w2cam": Pose.from_4x4mat(T),
            "camera": camera,
            "image_size": image_size, #WxH
        }
        if image is not None:
            data["image"] = image
        
        data["scales"] = np.array([1.0, 1.0], dtype=np.float32)

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].clone()
                x, y = kpts[:, 0].clone(), kpts[:, 1].clone()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x

                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}
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
            # seq_map, idx0 = self.items[idx]
            if isinstance(idx, list):
                seq_map, img_name = idx
            else:
                seq_map, img_name = self.items[idx]

            data = self._read_view(seq_map, img_name)
            # data = self._read_view(seq_map, idx0)
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
    dataset = EndomapperRoma(conf)
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
