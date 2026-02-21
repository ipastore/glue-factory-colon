import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

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
scene_lists_path = Path(__file__).parent / "endomapper_dense_seq_lists"


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
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",
        # Training
        "train_split": "train_seqs.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "val_seqs.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
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
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}
        self.overlaps = {}
        self.cameras = {}
        self.camera_indices = {}
        self.valid = {}

        self.info_dir = self.root / self.conf.info_dir
        self.scenes = []
        for scene in scenes:
            path = self.info_dir / (scene + ".npz")
            try:
                with np.load(str(path), allow_pickle=True) as info:
                    self.images[scene] = info["image_paths"]
                    self.depths[scene] = info["depth_paths"]
                    self.poses[scene] = info["poses"]
                    self.intrinsics[scene] = info["intrinsics"]
                    self.overlaps[scene] = info["overlap_matrix"].astype(
                        np.float32, copy=False
                    )
                    self.cameras[scene] = info["cameras"]
                    self.camera_indices[scene] = info["camera_indices"]
            except Exception:
                logger.warning(
                    "Cannot load scene info for scene %s at %s.", scene, path
                )
                continue

            self.valid[scene] = self._compute_valid(scene)
            valid_count = int(self.valid[scene].sum())
            total_count = len(self.valid[scene])
            if valid_count == 0:
                logger.warning("Skipping %s: no valid frames after filtering.", scene)
                continue
            if valid_count < total_count:
                logger.info(
                    "Scene %s: using %d/%d frames with required files.",
                    scene,
                    valid_count,
                    total_count,
                )
            self.scenes.append(scene)

        if len(self.scenes) == 0:
            raise ValueError(
                f"No valid EndomapperDense scenes for split '{split}'. "
                f"Check scene list in {scene_lists_path} and NPZ files in {self.info_dir}."
            )

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def _compute_valid(self, scene):
        n = len(self.images[scene])
        valid = np.ones(n, dtype=bool)
        if self.conf.read_image:
            image_exists = np.fromiter(
                ((self.root / str(path)).exists() for path in self.images[scene]),
                dtype=bool,
                count=n,
            )
            valid &= image_exists
        if self.conf.read_depth:
            depth_exists = np.fromiter(
                ((self.root / str(path)).exists() for path in self.depths[scene]),
                dtype=bool,
                count=n,
            )
            valid &= depth_exists
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

    def _load_camera(self, scene, idx, K):
        try:
            cam_idx = int(np.asarray(self.camera_indices[scene][idx]).item())
            return Camera.from_npz(self.cameras[scene][cam_idx]).float()
        except Exception:
            return Camera.from_calibration_matrix(K).float()


    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []
        split = self.split
        num_per_scene = self.conf[self.split + "_num_per_scene"]
        if isinstance(num_per_scene, Iterable):
            num_pos, num_neg = num_per_scene
        else:
            num_pos = num_per_scene
            num_neg = None
        if split != "train" and self.conf[split + "_pairs"] is not None:
            assert num_pos is None
            assert num_neg is None
            assert self.conf.views == 2
            pairs_path = scene_lists_path / self.conf[split + "_pairs"]
            for line in pairs_path.read_text().strip().splitlines():
                im0, im1 = line.split()
                scene = im0.split("/")[0]
                if scene not in self.images:
                    # Legacy strict behavior:
                    # assert scene in self.images
                    continue
                if im1.split("/")[0] != scene:
                    # Legacy strict behavior:
                    # assert im1.split("/")[0] == scene
                    continue
                im0, im1 = [self.conf.image_subpath + im for im in [im0, im1]]
                if im0 not in self.images[scene] or im1 not in self.images[scene]:
                    # Legacy strict behavior:
                    # assert im0 in self.images[scene]
                    # assert im1 in self.images[scene]
                    continue
                idx0 = np.where(self.images[scene] == im0)[0]
                idx1 = np.where(self.images[scene] == im1)[0]
                if len(idx0) == 0 or len(idx1) == 0:
                    continue
                if not (self.valid[scene][idx0[0]] and self.valid[scene][idx1[0]]):
                    continue
                self.items.append((scene, idx0[0], idx1[0], 1.0))
        elif self.conf.views == 1:
            for scene in self.scenes:
                # Legacy behavior (preprocessed arrays used None markers):
                # valid = (self.images[scene] != None) | (  # noqa: E711
                #     self.depths[scene] != None  # noqa: E711
                # )
                ids = np.where(self.valid[scene])[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)
        else:
            for scene in self.scenes:
                # Legacy behavior (None-marker filtering):
                # valid = (self.images[scene] != None) & (  # noqa: E711
                #     self.depths[scene] != None  # noqa: E711
                # )
                valid = self.valid[scene]
                if not np.any(valid):
                    continue
                ind = np.where(valid)[0]
                mat = self.overlaps[scene][valid][:, valid]
                if mat.size == 0:
                    continue

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
                    has_enough_samples = [len(p) >= num_per_bin * 2 for p in pairs_all]
                    num_per_bin_2 = num_pos // max(1, sum(has_enough_samples))
                    pairs = []
                    for pairs_bin, keep in zip(pairs_all, has_enough_samples):
                        if keep:
                            sampled = sample_n(pairs_bin, num_per_bin_2, seed)
                            if len(sampled) > 0:
                                pairs.append(sampled)
                    if len(pairs) == 0:
                        continue
                    pairs = np.concatenate(pairs, 0)
                else:
                    pairs = (mat > self.conf.min_overlap) & (
                        mat <= self.conf.max_overlap
                    )
                    pairs = np.stack(np.where(pairs), -1)
                    if len(pairs) == 0:
                        continue

                pairs = [(scene, ind[i], ind[j], mat[i, j]) for i, j in pairs]
                if num_neg is not None:
                    neg_pairs = np.stack(np.where(mat <= 0.0), -1)
                    neg_pairs = sample_n(neg_pairs, num_neg, seed)
                    pairs += [(scene, ind[i], ind[j], mat[i, j]) for i, j in neg_pairs]
                self.items.extend(pairs)
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]

        K = self.intrinsics[scene][idx].astype(np.float32, copy=True)
        T = self.poses[scene][idx].astype(np.float32, copy=False)
        camera = self._load_camera(scene, idx, K)

        if self.conf.read_image:
            img = load_image(path, self.conf.grayscale)
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
            depth_path = self.root / str(self.depths[scene][idx])
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

        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = torch.rot90(img, k=-k, dims=[1, 2])
                if self.conf.read_depth:
                    depth = torch.rot90(depth, k=-k, dims=[1, 2]).clone()
                K = rotate_intrinsics(K, img.shape, k + 2)
                camera = self._rotate_camera(camera, K, img.shape[-2:])
                T = rotate_pose_inplane(T, k + 2)

        name = path.name

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
                0
            ]
        camera = camera.scale(data["scales"])

        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(T),
            "depth": depth,
            "camera": camera,
            **data,
        }

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
        "num_workers": 0,
        "prefetch_factor": None,
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
