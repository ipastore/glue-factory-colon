import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Dict


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_heatmaps, plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane

logger = logging.getLogger(__name__)
seq_lists_path = Path(__file__).parent / "endomapper_dense_seq_lists"


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class EndomapperDense(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "endomapper_dense/",
        "depth_subpath": "depth_undistorted/",
        "specular_subpath": "specular_undistorted/",
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",
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
        "views": 2,
        "min_overlap": 0.3,
        "max_overlap": 1.0,
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,
        # image options
        "read_depth": True,
        "read_specular_mask": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
            path_not_found = DATA_PATH / conf.data_dir
            raise FileExistsError(
                f"Endomapper Dense Dataset not found in {path_not_found}"
            )

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            raise NotImplementedError(
                "Triplet Dataset not implemented for Endomapper Dense"
            )
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

        npz_dir = self.root / self.conf.info_dir
        for seq_map in seqs_maps:
            matching_files = sorted(npz_dir.glob(f"{seq_map}.npz"))
            if not matching_files:
                raise FileExistsError(f"No maps found for sequence {seq_map} in {npz_dir}")
                

        logger.info(f"Found {len(seqs_maps)} maps from for {split} split")


        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.specular_masks = {}
        self.cameras = {}

        self.seq: Dict[str, str] = {}
        self.map_id: Dict[str, str] = {}
        self.image_names: Dict[str, np.ndarray] = {}
        self.image_sizes: Dict[str, np.ndarray] = {}
        self.camera_ids: Dict[str, np.ndarray] = {}
        self.camera_indices: Dict[str, np.ndarray] = {}
        self.poses: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.valid: Dict[str, np.ndarray] = {}
        self.overlap_matrix: Dict[str, np.ndarray] = {}

        self.info_dir = self.root / self.conf.info_dir
        self.seqs_maps = []
        for seq_map in seqs_maps:
            path = self.info_dir / (seq_map + ".npz")
            try:
                with np.load(str(path), allow_pickle=True) as info:
                    self.images[seq_map] = info["image_paths"]
                    self.depths[seq_map] = info["depth_paths"]
                    self.specular_masks[seq_map] = info["specular_mask_paths"]
                    self.cameras[seq_map] = info["cameras"]

                    self.image_names[seq_map] = info["image_names"]
                    self.image_sizes[seq_map] = info["image_sizes"]
                    self.camera_ids[seq_map] = info["camera_ids"]
                    self.camera_indices[seq_map] = info["camera_indices"]
                    self.poses[seq_map] = info["poses"]
                    self.intrinsics[seq_map] = info["intrinsics"]
                    self.map_id[seq_map] = str(np.asarray(info["map_id"]).item())
                    self.seq[seq_map] = str(np.asarray(info["seq"]).item())
                    self.overlap_matrix[seq_map] = info["overlap_matrix"].astype(
                        np.float32, copy=False
                    )

            except Exception:
                logger.warning(
                    "Cannot load seq_map info for seq_map %s at %s.", seq_map, path
                )
                continue
            #HARDCODE safety net for loading depths+images from ongoing dataset
            self.valid[seq_map] = self._compute_valid(seq_map)
            valid_count = int(self.valid[seq_map].sum())
            total_count = len(self.valid[seq_map])
            if valid_count == 0:
                logger.warning("Skipping %s: no valid frames after filtering.", seq_map)
                continue
            if valid_count < total_count:
                logger.info(
                    "Seq_map %s: using %d/%d frames with required files.",
                    seq_map,
                    valid_count,
                    total_count,
                )
            self.seqs_maps.append(seq_map)
            #HARDCODE safety net for loading depths+images from ongoing dataset


        if len(self.seqs_maps) == 0:
            raise ValueError(
                f"No valid EndomapperDense seq_map for split '{split}'. "
                f"Check seq_map list in {seq_lists_path} and NPZ files in {self.info_dir}."
            )

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def _compute_valid(self, seq_map):
        n = len(self.images[seq_map])
        valid = np.ones(n, dtype=bool)
        if self.conf.read_image:
            image_exists = np.fromiter(
                ((self.root / str(path)).exists() for path in self.images[seq_map]),
                dtype=bool,
                count=n,
            )
            valid &= image_exists
        if self.conf.read_depth:
            depth_exists = np.fromiter(
                ((self.root / str(path)).exists() for path in self.depths[seq_map]),
                dtype=bool,
                count=n,
            )
            valid &= depth_exists
        if self.conf.read_specular_mask:
            specular_exists = np.fromiter(
                (
                    (self.root / str(path)).exists()
                    for path in self.specular_masks[seq_map]
                ),
                dtype=bool,
                count=n,
            )
            valid &= specular_exists
        return valid

    @staticmethod
    def _rotate_camera(camera, K, image_shape):
        h, w = image_shape
        data = camera._data.clone()
        data[0] = float(w)
        data[1] = float(h)
        data[2] = float(K[0, 0])
        data[3] = float(K[1, 1])
        data[4] = float(K[0, 2])
        data[5] = float(K[1, 2])
        rotated = Camera(data)
        rotated.model = camera.model
        return rotated.float()

    def _load_camera(self, seq_map, idx):
        cam_idx = int(np.asarray(self.camera_indices[seq_map][idx]).item())
        return Camera.from_npz(self.cameras[seq_map][cam_idx]).float()

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
        if split != "train" and self.conf[split + "_pairs"] is not None:
            assert num_pos is None
            assert num_neg is None
            assert self.conf.views == 2
            pairs_path = seq_lists_path / self.conf[split + "_pairs"]
            for line in pairs_path.read_text().strip().splitlines():
                #TODO adapt to new val_pairs.txt. Adapted from endomapper.py
                seq = line.split("/")[0]
                im0_name = str(line.split("/")[1].split("_")[1].strip(".png"))
                im1_name = str(line.split("/")[1].split("_")[-1].strip(".png"))
                if seq not in self.image_names:                    
                    # Legacy strict behavior:
                    # assert scene in self.images
                    continue
                if im0_name not in self.image_names[seq] or im1_name not in self.image_names[seq]:
                    # Legacy strict behavior:
                    # assert im0 in self.images[scene]
                    # assert im1 in self.images[scene]
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
                ids = [(seq_map, self.image_names[seq_map][i]) for i in ids]
                self.items.extend(ids)
        else:
            for seq_map in self.seqs_maps:
                # Legacy behavior (None-marker filtering):
                # valid = (self.images[scene] != None) & (  # noqa: E711
                #     self.depths[scene] != None  # noqa: E711
                # )

                #HARDCODE SAFETY NET, leave mat
                valid = self.valid[seq_map]
                if not np.any(valid):
                    continue
                mat = self.overlap_matrix[seq_map].astype(np.float32, copy=True)
                invalid = ~valid
                mat[invalid, :] = -1.0
                mat[:, invalid] = -1.0
                #HARDCODE SAFETY NET


                if num_pos is not None:
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
                pairs = [
                    (seq_map, image_names[i], image_names[j], mat[i, j]) for i, j in pairs
                ]
                if num_neg is not None:
                    neg_pairs = np.stack(np.where(mat <= 0.0), -1)
                    neg_pairs = sample_n(neg_pairs, num_neg, seed)
                    pairs += [
                        (seq_map, image_names[i], image_names[j], mat[i, j])
                        for i, j in neg_pairs
                    ]
                self.items.extend(pairs)
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, seq_map, image_name):
        image_names = self.image_names[seq_map]
        if isinstance(image_name, (int, np.integer)):
            idx = int(image_name)
            image_name = image_names[idx]
        else:
            idx = np.where(image_names == image_name)[0][0]
        T = self.poses[seq_map][idx].astype(np.float32, copy=False)
        K = self.intrinsics[seq_map][idx].astype(np.float32, copy=True)
        camera = self._load_camera(seq_map, idx)

        if self.conf.read_image:
            img_path = self.root / str(self.images[seq_map][idx])
            img = load_image(img_path)
        else:
            cam_size = camera.size.cpu().numpy().astype(np.int32)
            size = int(cam_size[1]), int(cam_size[0])
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()

        raw_image_shape = tuple(img.shape[-2:])
        img, crop_left_top = self.preprocessor.crop_endomapper_dense(img)
        crop_left, crop_top = crop_left_top
        K[0, 2] -= crop_left
        K[1, 2] -= crop_top
        camera = camera.crop(crop_left_top, (img.shape[-1], img.shape[-2]))

        if self.conf.read_depth:
            depth_path = self.root / str(self.depths[seq_map][idx])
            try:
                with np.load(str(depth_path)) as depth_data:
                    depth = depth_data["depth"].astype(np.float32, copy=False)
                    mask = depth_data["mask"].astype(bool, copy=False)
            except Exception as e:
                raise OSError(f"Failed reading depth file: {depth_path}") from e

            if depth.shape != mask.shape:
                raise ValueError(f"Depth/mask shape mismatch in {depth_path}.")
            depth = np.where(mask, depth, 0.0).astype(np.float32, copy=False)
            depth = torch.from_numpy(depth)[None]

            if tuple(depth.shape[-2:]) == raw_image_shape:
                depth, depth_left_top = self.preprocessor.crop_endomapper_dense(depth)
                if depth_left_top != crop_left_top:
                    raise ValueError(
                        f"Unexpected crop offset for depth {depth_path}: "
                        f"{depth_left_top} vs {crop_left_top}."
                    )
            elif tuple(depth.shape[-2:]) != tuple(img.shape[-2:]):
                raise ValueError(
                    f"Depth shape mismatch for {depth_path}: {depth.shape[-2:]} "
                    f"vs image {img.shape[-2:]}."
                )
            assert depth.shape[-2:] == img.shape[-2:]
        else:
            depth = None
        specular_mask = None
        if self.conf.read_specular_mask:
            specular_path = self.root / str(self.specular_masks[seq_map][idx])
            try:
                with np.load(str(specular_path)) as spec_data:
                    if "mask_packbits" in spec_data and "mask_shape" in spec_data:
                        packed = spec_data["mask_packbits"]
                        h, w = spec_data["mask_shape"].astype(np.int64).tolist()
                        flat = np.unpackbits(packed, count=int(h * w))
                        specular_mask_np = flat.reshape((h, w)).astype(bool, copy=False)
                    else:
                        raise KeyError(f"Specular mask array not found in {specular_path}.")
            except Exception as e:
                raise OSError(
                    f"Failed reading specular mask file: {specular_path}"
                ) from e

            specular_mask = torch.from_numpy(specular_mask_np)[None]
            if tuple(specular_mask.shape[-2:]) == raw_image_shape:
                specular_mask, spec_left_top = self.preprocessor.crop_endomapper_dense(
                    specular_mask
                )
                if spec_left_top != crop_left_top:
                    raise ValueError(
                        f"Unexpected crop offset for specular mask {specular_path}: "
                        f"{spec_left_top} vs {crop_left_top}."
                    )
            elif tuple(specular_mask.shape[-2:]) != tuple(img.shape[-2:]):
                raise ValueError(
                    f"Specular mask shape mismatch for {specular_path}: "
                    f"{specular_mask.shape[-2:]} vs image {img.shape[-2:]}."
                )
            assert specular_mask.shape[-2:] == img.shape[-2:]

        #Not tested
        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = torch.rot90(img, k=-k, dims=[1, 2])
                if self.conf.read_depth:
                    depth = torch.rot90(depth, k=-k, dims=[1, 2]).clone()
                if specular_mask is not None:
                    specular_mask = torch.rot90(
                        specular_mask.float(), k=-k, dims=[1, 2]
                    ) > 0.5
                K = rotate_intrinsics(K, img.shape, k + 2)
                camera = self._rotate_camera(camera, K, img.shape[-2:])
                T = rotate_pose_inplane(T, k + 2)

        data = self.preprocessor(img)
        if depth is not None:
            depth = self.preprocessor(depth, interpolation="nearest")["image"][0]
        if specular_mask is not None:
            specular_mask = (
                self.preprocessor(
                    specular_mask.float(), interpolation="nearest"
                )["image"][0]
                > 0.5
            )
        camera = camera.scale(data["scales"])

        data = {
            "name": image_name,
            "seq_map": seq_map,
            "T_w2cam": Pose.from_4x4mat(T),
            "depth": depth,
            "camera": camera,
            "specular_mask": specular_mask,
            **data,
        }

        #Not tested
        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
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
                scene, idx0, idx1, overlap = idx
            else:
                scene, idx0, idx1, overlap = self.items[idx]
            data0 = self._read_view(scene, idx0)
            data1 = self._read_view(scene, idx1)
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["overlap_0to1"] = overlap
            data["name"] = f"{scene}/{data0['name']}_{data1['name']}"
        else:
            assert self.conf.views == 1
            scene, idx0 = self.items[idx]
            data = self._read_view(scene, idx0)
        data["scene"] = scene
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
        "num_workers": 1,
        "prefetch_factor": 2,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = EndomapperDense(conf)
    loader = dataset.get_data_loader(args.split)
    logger.info("The dataset has %d elements.", len(loader))

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
