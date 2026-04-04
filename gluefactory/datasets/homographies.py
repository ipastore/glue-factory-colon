"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points,
)
from ..models.cache_loader import CacheLoader, pad_local_features
from ..settings import DATA_PATH
from ..utils.image import load_image, read_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


class HomographyDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "revisitop1m",  # the top-level directory
        "image_dir": "jpg/",  # the subdirectory with the images
        "image_list": "revisitop1m.txt",  # optional: list or filename of list
        "check_file_exists": False,  # check if the image exists
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        # splits
        "train_size": 100,
        "val_size": 10,
        "test_size": 0,
        "shuffle_seed": 0,  # or None to skip
        # image loading
        "grayscale": False,
        "triplet": False,
        "right_only": False,  # image0 is orig (rescaled), image1 is right
        "left_view_difficulty": None,
        "right_view_difficulty": None,
        "reseed": False,
        "test_seed": 0,
        "test_sequences": None,
        "test_homography_levels": None,
        "test_photometric_levels": None,
        "test_source_dir": None,
        "test_image_list": None,
        "save_dataset": False,
        "read_dataset_from_disk": False,
        "saved_dataset_dir": "endopatches1800",
        "crop_vignette": False,  # For endomapper. Since we are using a patch smaller than the original image size, we take advantage of resize to crop the vignette
        "vignette_crop_coords": None,  # [x1, x2, y1, y2] or None for auto-center
        "homography": {
            "difficulty": 0.8,
            "translation": 1.0,
            "max_angle": 60,
            "n_angles": 10,
            "patch_shape": [640, 480],
            "min_convexity": 0.05,
        },
        "photometric": {
            "name": "dark",
            "p": 0.75,
            # 'difficulty': 1.0,  # currently unused
        },
        # feature loading
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
            "thresh": 0.0,
            "max_num_keypoints": -1,
            "force_num_keypoints": False,
        },
    }

    def _init(self, conf):
        if conf.read_dataset_from_disk and conf.test_source_dir is not None:
            raise ValueError(
                "test_source_dir and read_dataset_from_disk are mutually exclusive."
            )

        data_dir = DATA_PATH / conf.data_dir
        if not data_dir.exists():
            if conf.data_dir == "revisitop1m":
                logger.info("Downloading the revisitop1m dataset.")
                self.download_revisitop1m()
            else:
                raise FileNotFoundError(data_dir)
        
        #Handle split-based folder structure (endomapper)
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        test_dir = data_dir / "test"

        if train_dir.exists() and val_dir.exists():
            # Load images from train and val folders separately
            train_images = self._load_images_from_folder(train_dir, conf)
            val_images = self._load_images_from_folder(val_dir, conf)
            test_images = (
                self._load_test_images(test_dir, conf) if test_dir.exists() else []
            )
           
            if conf.shuffle_seed is not None:
                np.random.RandomState(conf.shuffle_seed).shuffle(train_images)
                np.random.RandomState(conf.shuffle_seed + 1).shuffle(val_images)
            if conf.train_size > 0:
                train_images = train_images[: conf.train_size]
            if conf.val_size > 0:
                val_images = val_images[: conf.val_size]
            
            self.images = {
                "train": train_images,
                "val": val_images,
                "test": test_images,
            }
        else:
            #Fallback to original behavior (revisitop1m)
            image_dir = data_dir / conf.image_dir
            images = []
            if conf.image_list is None:
                glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
                for g in glob:
                    images += list(image_dir.glob("**/" + g))
                if len(images) == 0:
                    raise ValueError(f"Cannot find any image in folder: {image_dir}.")
                images = [i.relative_to(image_dir).as_posix() for i in images]
                images = sorted(images)  # for deterministic behavior
                logger.info("Found %d images in folder.", len(images))
            elif isinstance(conf.image_list, (str, Path)):
                image_list = data_dir / conf.image_list
                if not image_list.exists():
                    raise FileNotFoundError(f"Cannot find image list {image_list}.")
                images = image_list.read_text().rstrip("\n").split("\n")
                for image in images:
                    if self.conf.check_file_exists and not (image_dir / image).exists():
                        raise FileNotFoundError(image_dir / image)
                logger.info("Found %d images in list file.", len(images))
            elif isinstance(conf.image_list, omegaconf.listconfig.ListConfig):
                images = conf.image_list.to_container()
                for image in images:
                    if self.conf.check_file_exists and not (image_dir / image).exists():
                        raise FileNotFoundError(image_dir / image)
            else:
                raise ValueError(conf.image_list)

            if conf.shuffle_seed is not None:
                np.random.RandomState(conf.shuffle_seed).shuffle(images)
            train_images = images[: conf.train_size]
            val_images = images[conf.train_size : conf.train_size + conf.val_size]
            self.images = {"train": train_images, "val": val_images, "test": []}
    
    #For endomapper
    def _load_images_from_folder(self, folder, conf):
        """Load images from a specific folder using glob patterns."""
        images = []
        glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
        for g in glob:
            images += list(folder.glob("**/" + g))
        if len(images) == 0:
            logger.warning(f"No images found in folder: {folder}")
            return []
        # Make paths relative to the split folder for compatibility
        images = [i.relative_to(folder).as_posix() for i in images]
        images = sorted(images)
        logger.info(f"Found {len(images)} images in {folder.name}/")
        return images

    def _load_test_images(self, folder, conf):
        images = self._load_images_from_folder(folder, conf)
        if conf.test_sequences is None:
            return images
        keep_sequences = set(conf.test_sequences)
        filtered = [
            name for name in images if self._parse_test_sequence(name) in keep_sequences
        ]
        logger.info(
            "Filtered %d/%d test images for benchmark sequences.",
            len(filtered),
            len(images),
        )
        return filtered

    @staticmethod
    def _parse_test_sequence(name):
        stem = Path(name).stem
        prefix, sep, suffix = stem.rpartition("_")
        if not sep or not suffix.isdigit():
            return None
        return prefix

    @staticmethod
    def _has_test_benchmark_schedule(conf):
        return (
            conf.test_size > 0
            and conf.test_sequences is not None
            and conf.test_homography_levels is not None
            and conf.test_photometric_levels is not None
        )

    def download_revisitop1m(self):
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "revisitop1m_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        image_dir = tmp_dir / self.conf.image_dir
        image_dir.mkdir(exist_ok=True, parents=True)
        num_files = 100
        url_base = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/"
        list_name = "revisitop1m.txt"
        torch.hub.download_url_to_file(url_base + list_name, tmp_dir / list_name)
        for n in tqdm(range(num_files), position=1):
            tar_name = "revisitop1m.{}.tar.gz".format(n + 1)
            tar_path = image_dir / tar_name
            torch.hub.download_url_to_file(url_base + "jpg/" + tar_name, tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=image_dir)
            tar_path.unlink()
        shutil.move(tmp_dir, data_dir)

    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_names, split):
        self.conf = conf
        self.split = split
        self.image_names = np.array(image_names)
        self.test_items = None
        self.saved_root = None

        # Handle split-based folder structure (endomapper)
        base_dir = DATA_PATH / conf.data_dir
        if (base_dir / split).exists():
            self.image_dir = base_dir / split
        else:
            self.image_dir = base_dir / conf.image_dir

        aug_conf = conf.photometric
        aug_name = aug_conf.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'
        self.photo_augment = augmentations[aug_name](aug_conf)
        self.left_augment = (
            IdentityAugmentation() if conf.right_only else self.photo_augment
        )
        self.img_to_tensor = IdentityAugmentation()

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        if split == "test":
            if conf.read_dataset_from_disk:
                self.saved_root = DATA_PATH / conf.saved_dataset_dir
                self.test_items = self._load_saved_test_items()
            elif HomographyDataset._has_test_benchmark_schedule(conf):
                self.test_items = self._build_test_schedule()
                if conf.save_dataset:
                    self._save_test_dataset()
            if self.test_items is not None:
                self.image_names = np.array([item["name"] for item in self.test_items])

    def _build_test_schedule(self):
        images_by_sequence = {}
        for name in self.image_names.tolist():
            seq = HomographyDataset._parse_test_sequence(name)
            if seq is None:
                continue
            images_by_sequence.setdefault(seq, []).append(name)

        homography_levels = list(self.conf.test_homography_levels)
        photometric_levels = list(self.conf.test_photometric_levels)
        num_variants = len(homography_levels) * len(photometric_levels)
        if self.conf.test_size <= 0 or self.conf.test_size % num_variants != 0:
            raise ValueError(
                "test_size must be a positive multiple of "
                "len(test_homography_levels) * len(test_photometric_levels)."
            )
        source_names = self._select_test_sources(images_by_sequence)
        items = []
        for source_idx, image_name in enumerate(source_names):
            seq = HomographyDataset._parse_test_sequence(image_name) or ""
            source_dir = Path(image_name).stem
            for homo_idx, homo_level in enumerate(homography_levels):
                homography_conf = self._resolve_homography_level(homo_level)
                for photo_idx, photo_level in enumerate(photometric_levels):
                    photometric_conf = self._resolve_photometric_level(photo_level)
                    variant_name = f"h{homo_idx}_p{photo_idx}.png"
                    global_idx = len(items)
                    items.append(
                        {
                            "idx": global_idx,
                            "name": f"{source_dir}/{variant_name}",
                            "source_name": image_name,
                            "source_dir": source_dir,
                            "scene": seq,
                            "seed": self.conf.test_seed + global_idx,
                            "variant_name": variant_name,
                            "homography_file": (
                                f"H_{source_dir}_{Path(variant_name).stem}.txt"
                            ),
                            "homography_idx": homo_idx,
                            "homography_level": float(
                                homography_conf["difficulty"]
                            ),
                            "photometric_idx": photo_idx,
                            "photometric_level": float(photometric_conf["p"]),
                            "homography_conf": homography_conf,
                            "photometric_conf": photometric_conf,
                            "source_idx": source_idx,
                        }
                    )
        return items

    def _select_test_sources(self, images_by_sequence):
        if self.conf.test_source_dir is not None:
            source_names = self._read_source_dir(self.conf.test_source_dir)
            missing = [
                name for name in source_names if name not in self.image_names.tolist()
            ]
            if missing:
                raise ValueError(
                    f"Unknown test images in {self.conf.test_source_dir}: {missing[:5]}"
                )
        elif self.conf.test_image_list is not None:
            source_names = self._read_source_list(self.conf.test_image_list)
            missing = [
                name
                for name in source_names
                if name not in self.image_names.tolist()
            ]
            if missing:
                raise ValueError(
                    f"Unknown test images in {self.conf.test_image_list}: {missing[:5]}"
                )
        else:
            sequences = list(self.conf.test_sequences)
            num_variants = (
                len(self.conf.test_homography_levels)
                * len(self.conf.test_photometric_levels)
            )
            num_sources = self.conf.test_size // num_variants
            if num_sources % len(sequences) != 0:
                raise ValueError(
                    "test_size must select the same number of sources per sequence."
                )
            per_sequence = num_sources // len(sequences)
            source_names = []
            for seq_idx, seq in enumerate(sequences):
                seq_images = sorted(images_by_sequence.get(seq, []))
                if len(seq_images) < per_sequence:
                    raise ValueError(
                        f"Not enough test images in sequence {seq}: "
                        f"need {per_sequence}, found {len(seq_images)}."
                    )
                rng = np.random.RandomState(self.conf.test_seed + seq_idx)
                order = rng.permutation(len(seq_images))[:per_sequence]
                source_names.extend([seq_images[i] for i in order])

        expected_sources = self.conf.test_size // (
            len(self.conf.test_homography_levels)
            * len(self.conf.test_photometric_levels)
        )
        if len(source_names) != expected_sources:
            raise ValueError(
                f"Expected {expected_sources} test sources, found {len(source_names)}."
            )
        if len(set(source_names)) != len(source_names):
            raise ValueError("test source list contains duplicates.")
        return source_names

    def _read_source_dir(self, path_conf):
        path = Path(path_conf)
        if not path.is_absolute():
            path = DATA_PATH / path
        if not path.exists():
            raise FileNotFoundError(f"Cannot find test source dir {path_conf}.")

        source_names = []
        glob = [self.conf.glob] if isinstance(self.conf.glob, str) else self.conf.glob
        for seq in self.conf.test_sequences:
            seq_dir = path / seq
            if not seq_dir.exists():
                raise FileNotFoundError(f"Cannot find sequence dir {seq_dir}.")
            seq_images = []
            for g in glob:
                seq_images += list(seq_dir.glob(g))
            if len(seq_images) == 0:
                raise ValueError(f"No images found in {seq_dir}.")
            source_names.extend(sorted(image.name for image in seq_images))
        return source_names

    def _read_source_list(self, path_conf):
        path = Path(path_conf)
        if not path.is_absolute():
            candidates = [
                DATA_PATH / self.conf.data_dir / path,
                DATA_PATH / path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise FileNotFoundError(f"Cannot find test image list {path_conf}.")
        return [line for line in path.read_text().splitlines() if line.strip()]

    def _resolve_homography_level(self, level):
        conf = omegaconf.OmegaConf.to_container(self.conf.homography, resolve=True)
        if isinstance(level, (int, float)):
            conf["difficulty"] = float(level)
        else:
            conf.update(omegaconf.OmegaConf.to_container(level, resolve=True))
        return conf

    def _resolve_photometric_level(self, level):
        conf = omegaconf.OmegaConf.to_container(self.conf.photometric, resolve=True)
        if isinstance(level, (int, float)):
            conf["p"] = float(level)
        else:
            conf.update(omegaconf.OmegaConf.to_container(level, resolve=True))
        return conf

    def _make_augmentations(self, photometric_conf):
        photo_augment = augmentations[photometric_conf["name"]](photometric_conf)
        left_augment = (
            IdentityAugmentation()
            if self.conf.right_only
            else augmentations[photometric_conf["name"]](photometric_conf)
        )
        return left_augment, photo_augment

    def _transform_keypoints(self, features, data):
        """Transform keypoints by a homography, threshold them,
        and potentially keep only the best ones."""
        # Warp points
        features["keypoints"] = warp_points(
            features["keypoints"], data["H_"], inverse=False
        )
        h, w = data["image"].shape[1:3]
        valid = (
            (features["keypoints"][:, 0] >= 0)
            & (features["keypoints"][:, 0] <= w - 1)
            & (features["keypoints"][:, 1] >= 0)
            & (features["keypoints"][:, 1] <= h - 1)
        )
        features["keypoints"] = features["keypoints"][valid]

        # Threshold
        if self.conf.load_features.thresh > 0:
            valid = features["keypoint_scores"] >= self.conf.load_features.thresh
            features = {k: v[valid] for k, v in features.items()}

        # Get the top keypoints and pad
        n = self.conf.load_features.max_num_keypoints
        if n > -1:
            inds = np.argsort(-features["keypoint_scores"])
            features = {k: v[inds[:n]] for k, v in features.items()}

            if self.conf.load_features.force_num_keypoints:
                features = pad_local_features(
                    features, self.conf.load_features.max_num_keypoints
                )

        return features

    def __getitem__(self, idx):
        if self.saved_root is not None:
            return self.getitem_saved(idx)
        seed = None
        if self.test_items is not None:
            seed = self.test_items[idx]["seed"]
        elif self.conf.reseed:
            seed = self.conf.seed + idx
        if seed is not None:
            with fork_rng(seed, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def _read_view(
        self, img, H_conf, ps, photo_augment, left_augment, left=False, seed=None
    ):
        data = sample_homography(img, H_conf, ps)
        if left:
            data["image"] = left_augment(data["image"], return_tensor=True, seed=seed)
        else:
            data["image"] = photo_augment(data["image"], return_tensor=True, seed=seed)

        gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        if self.conf.grayscale:
            data["image"] = (data["image"] * gs).sum(0, keepdim=True)

        data["scales"] = np.array([1.0, 1.0], dtype=np.float32)
        data["image_size"] = np.array(
            data["image"].shape[-2:][::-1],
            dtype=np.float32,
        )

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            features = self._transform_keypoints(features, data)
            data["cache"] = features
        return data

    @staticmethod
    def _save_png(path, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3:
            image = image.transpose(1, 2, 0)
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            ok = cv2.imwrite(str(path), image)
        elif image.shape[-1] == 1:
            ok = cv2.imwrite(str(path), image[..., 0])
        else:
            ok = cv2.imwrite(str(path), image[..., ::-1])
        if not ok:
            raise IOError(f"Could not write image to {path}.")

    @staticmethod
    def _write_homography(path, H):
        np.savetxt(path, H, fmt="%.18e")

    @staticmethod
    def _format_attrib_line(
        image_name,
        source_name,
        scene,
        kind,
        homography_idx,
        homography_level,
        photometric_idx,
        photometric_level,
    ):
        return (
            f"{image_name} {source_name} {scene} {kind} "
            f"{homography_idx} {homography_level:.6f} "
            f"{photometric_idx} {photometric_level:.6f}"
        )

    def _save_test_dataset(self):
        root = DATA_PATH / self.conf.saved_dataset_dir
        root.mkdir(parents=True, exist_ok=True)
        attribs = {}
        source_names = []
        seen_sources = set()
        for item in self.test_items:
            data = self.getitem(item["idx"])
            source_dir = root / item["source_dir"]
            source_dir.mkdir(parents=True, exist_ok=True)
            if item["source_name"] not in seen_sources:
                self._save_png(source_dir / "source.png", data["view0"]["image"])
                source_names.append(item["source_name"])
                attribs[item["source_dir"]] = [
                    self._format_attrib_line(
                        "source.png",
                        item["source_name"],
                        item["scene"],
                        "source",
                        -1,
                        0.0,
                        -1,
                        0.0,
                    )
                ]
                seen_sources.add(item["source_name"])
            self._save_png(
                source_dir / item["variant_name"],
                data["view1"]["image"],
            )
            self._write_homography(
                source_dir / item["homography_file"],
                data["H_0to1"],
            )
            attribs[item["source_dir"]].append(
                self._format_attrib_line(
                    item["variant_name"],
                    item["source_name"],
                    item["scene"],
                    "variant",
                    item["homography_idx"],
                    item["homography_level"],
                    item["photometric_idx"],
                    item["photometric_level"],
                )
            )

        for source_dir, lines in attribs.items():
            (root / source_dir / "attribs.txt").write_text("\n".join(lines) + "\n")
        (root / "source_names.txt").write_text("\n".join(source_names) + "\n")

    def _load_saved_test_items(self):
        root = self.saved_root
        source_file = root / "source_names.txt"
        if not source_file.exists():
            raise FileNotFoundError(
                f"Cannot find saved EndoPatches manifest at {source_file}."
            )
        source_names = [
            line for line in source_file.read_text().splitlines() if line.strip()
        ]
        items = []
        for source_name in source_names:
            source_dir = Path(source_name).stem
            attrib_file = root / source_dir / "attribs.txt"
            if not attrib_file.exists():
                raise FileNotFoundError(f"Cannot find attribs file {attrib_file}.")
            for line in attrib_file.read_text().splitlines():
                if not line.strip():
                    continue
                (
                    image_name,
                    source_name_line,
                    scene,
                    kind,
                    homography_idx,
                    homography_level,
                    photometric_idx,
                    photometric_level,
                ) = line.split(" ")
                if kind == "source":
                    continue
                items.append(
                    {
                        "idx": len(items),
                        "name": f"{source_dir}/{image_name}",
                        "scene": scene,
                        "source_name": source_name_line,
                        "source_dir": source_dir,
                        "variant_name": image_name,
                        "homography_file": (
                            f"H_{source_dir}_{Path(image_name).stem}.txt"
                        ),
                        "homography_idx": int(homography_idx),
                        "homography_level": float(homography_level),
                        "photometric_idx": int(photometric_idx),
                        "photometric_level": float(photometric_level),
                    }
                )
        return items

    def getitem_saved(self, idx):
        item = self.test_items[idx]
        source_dir = self.saved_root / item["source_dir"]
        source = load_image(source_dir / "source.png", grayscale=self.conf.grayscale)
        variant = load_image(
            source_dir / item["variant_name"], grayscale=self.conf.grayscale
        )
        H = np.loadtxt(source_dir / item["homography_file"]).astype(np.float32)
        scales = np.array([1.0, 1.0], dtype=np.float32)
        view0 = {
            "image": source,
            "image_size": np.array(source.shape[-2:][::-1], dtype=np.float32),
            "scales": scales,
        }
        view1 = {
            "image": variant,
            "image_size": np.array(variant.shape[-2:][::-1], dtype=np.float32),
            "scales": scales,
        }
        size = np.array(source.shape[-2:][::-1])
        return {
            "name": item["name"],
            "scene": item["scene"],
            "source_name": item["source_name"],
            "original_image_size": size,
            "scales": scales,
            "H_0to1": H,
            "idx": item["idx"],
            "view0": view0,
            "view1": view1,
        }

    def getitem(self, idx):
        item = None if self.test_items is None else self.test_items[idx]
        source_name = self.image_names[idx] if item is None else item["source_name"]
        img = read_image(self.image_dir / source_name, False)
        if img is None:
            logging.warning("Image %s could not be read.", source_name)
            img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
        img = img.astype(np.float32) / 255.0


        # For endomapper: apply center crop to remove vignette borders
        if self.conf.crop_vignette:
            if self.conf.vignette_crop_coords is not None:
                # Use fixed crop coordinates
                x1, x2, y1, y2 = self.conf.vignette_crop_coords
                img = img[y1:y2, x1:x2]
            else:
                # Fallback: no crop if coordinates not specified
                logger.warning("crop_vignette enabled but vignette_crop_coords not set")


        size = img.shape[:2][::-1]
        homography_conf = (
            omegaconf.OmegaConf.to_container(self.conf.homography, resolve=True)
            if item is None
            else dict(item["homography_conf"])
        )
        photometric_conf = (
            omegaconf.OmegaConf.to_container(self.conf.photometric, resolve=True)
            if item is None
            else dict(item["photometric_conf"])
        )
        ps = homography_conf["patch_shape"]

        if item is None:
            left_conf = dict(homography_conf)
            if self.conf.right_only:
                left_conf["difficulty"] = 0.0
            if self.conf.left_view_difficulty is not None:
                left_conf["difficulty"] = self.conf.left_view_difficulty
            right_conf = dict(homography_conf)
            if self.conf.right_view_difficulty is not None:
                right_conf["difficulty"] = self.conf.right_view_difficulty
            left_augment, photo_augment = self._make_augmentations(photometric_conf)
            data0 = self._read_view(
                img, left_conf, ps, photo_augment, left_augment, left=True
            )
            data1 = self._read_view(
                img, right_conf, ps, photo_augment, left_augment, left=False
            )
        else:
            with fork_rng(item["seed"] * 2, False):
                left_conf = dict(homography_conf)
                left_conf["difficulty"] = 0.0
                left_augment = IdentityAugmentation()
                photo_augment = augmentations[photometric_conf["name"]](
                    photometric_conf
                )
                data0 = self._read_view(
                    img,
                    left_conf,
                    ps,
                    photo_augment,
                    left_augment,
                    left=True,
                    seed=item["seed"] * 2,
                )
            with fork_rng(item["seed"] * 2 + 1, False):
                right_conf = dict(homography_conf)
                photo_augment = augmentations[photometric_conf["name"]](
                    photometric_conf
                )
                left_augment = IdentityAugmentation()
                data1 = self._read_view(
                    img,
                    right_conf,
                    ps,
                    photo_augment,
                    left_augment,
                    left=False,
                    seed=item["seed"] * 2 + 1,
                )

        H = compute_homography(data0["coords"], data1["coords"], [1, 1])

        data = {
            "name": source_name if item is None else item["name"],
            "scene": "" if item is None else item["scene"],
            "source_name": source_name,
            "original_image_size": np.array(size),
            "scales": np.array([1.0, 1.0], dtype=np.float32),
            "H_0to1": H.astype(np.float32),
            "idx": idx if item is None else item["idx"],
            "view0": data0,
            "view1": data1,
        }

        if self.conf.triplet:
            # Generate third image
            data2 = self._read_view(img, self.conf.homography, ps, left=False)
            H02 = compute_homography(data0["coords"], data2["coords"], [1, 1])
            H12 = compute_homography(data1["coords"], data2["coords"], [1, 1])

            data = {
                "H_0to2": H02.astype(np.float32),
                "H_1to2": H12.astype(np.float32),
                "view2": data2,
                **data,
            }

        return data

    def __len__(self):
        if self.test_items is not None:
            return len(self.test_items)
        return len(self.image_names)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = HomographyDataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
