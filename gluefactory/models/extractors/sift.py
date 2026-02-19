import warnings

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_grayscale
from packaging import version

try:
    import pycolmap
except ImportError:
    pycolmap = None

try:
    import cudasift_py
except ImportError:
    cudasift_py = None

from ..base_model import BaseModel
from ..utils.misc import pad_to_length


def filter_dog_point(points, scales, angles, image_shape, nms_radius, scores=None):
    h, w = image_shape
    ij = np.round(points - 0.5).astype(int).T[::-1]

    # Remove duplicate points (identical coordinates).
    # Pick highest scale or score
    s = scales if scores is None else scores
    buffer = np.zeros((h, w))
    np.maximum.at(buffer, tuple(ij), s)
    keep = np.where(buffer[tuple(ij)] == s)[0]

    # Pick lowest angle (arbitrary).
    ij = ij[:, keep]
    buffer[:] = np.inf
    o_abs = np.abs(angles[keep])
    np.minimum.at(buffer, tuple(ij), o_abs)
    mask = buffer[tuple(ij)] == o_abs
    ij = ij[:, mask]
    keep = keep[mask]

    if nms_radius > 0:
        # Apply NMS on the remaining points
        buffer[:] = 0
        buffer[tuple(ij)] = s[keep]  # scores or scale

        local_max = torch.nn.functional.max_pool2d(
            torch.from_numpy(buffer).unsqueeze(0),
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(0)
        is_local_max = buffer == local_max.numpy()
        keep = keep[is_local_max[tuple(ij)]]
    return keep


def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def run_opencv_sift(features: cv2.Feature2D, image: np.ndarray) -> np.ndarray:
    """
    Detect keypoints using OpenCV Detector.
    Optionally, perform description.
    Args:
        features: OpenCV based keypoints detector and descriptor
        image: Grayscale image of uint8 data type
    Returns:
        keypoints: 1D array of detected cv2.KeyPoint
        scores: 1D array of responses
        descriptors: 1D array of descriptors
    """
    detections, descriptors = features.detectAndCompute(image, None)
    points = np.array([k.pt for k in detections], dtype=np.float32)
    scores = np.array([k.response for k in detections], dtype=np.float32)
    scales = np.array([k.size for k in detections], dtype=np.float32)
    angles = np.deg2rad(np.array([k.angle for k in detections], dtype=np.float32))
    return points, scores, scales, angles, descriptors


class SIFT(BaseModel):
    default_conf = {
        "rootsift": True,
        "nms_radius": 0,  # None to disable filtering entirely.
        "max_num_keypoints": 4096,
        "backend": "opencv",  # in {opencv, pycolmap, pycolmap_cpu, pycolmap_cuda, py_cudasift, py_CudaSift, py_Cudasift}
        "detection_threshold": 0.0066667,  # from COLMAP
        "edge_threshold": 10,
        "first_octave": -1,  # only used by pycolmap, the default of COLMAP
        "num_octaves": 4,
        "init_blur": 1.0,  # used by py_cudasift
        "filter_kpts_with_wrapper": True,  # only used by py_cudasift
        "filter_with_scale_weighting": False,  # if true: rank by abs(score) * scale
        "extractor_channel": "grayscale",  # in {grayscale, red, green, blue}
        "sort_scales_low_to_large": False,  # False: large->small, True: small->large
        "force_num_keypoints": False,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        backend = self.conf.backend
        if backend.startswith("pycolmap"):
            if pycolmap is None:
                raise ImportError(
                    "Cannot find module pycolmap: install it with pip"
                    "or use backend=opencv."
                )
            
            pycolmap_version = version.parse(pycolmap.__version__)

            if pycolmap_version < version.parse("3.13.0"):
                # options = {
                #     "peak_threshold": self.conf.detection_threshold,
                #     "edge_threshold": self.conf.edge_threshold,
                #     "first_octave": self.conf.first_octave,
                #     "num_octaves": self.conf.num_octaves,
                #     "normalization": pycolmap.Normalization.L2,  # L1_ROOT is buggy.
                # }
                options = pycolmap.SiftExtractionOptions()
                options.peak_threshold = self.conf.detection_threshold
                options.edge_threshold = self.conf.edge_threshold
                options.first_octave = self.conf.first_octave
                options.num_octaves = self.conf.num_octaves
                options.normalization = pycolmap.Normalization.L2  # L1_ROOT is buggy.
                device = (
                    "auto" if backend == "pycolmap" else backend.replace("pycolmap_", "")
                )
                if (
                backend == "pycolmap_cpu" or not pycolmap.has_cuda
                ) and pycolmap_version < version.parse("0.5.0"):
                    warnings.warn(
                        "The pycolmap CPU SIFT is buggy in version < 0.5.0, "
                        "consider upgrading pycolmap or use the CUDA version.",
                        stacklevel=1,
                    )
                else: 
                    # options["max_num_features"] = self.conf.max_num_keypoints
                    options.max_num_features = self.conf.max_num_keypoints

            else: 
                options = pycolmap.FeatureExtractionOptions()
                sift_opts = options.sift
                
                # Set SIFT-specific options via the nested .sift attribute
                # prefer the nested .sift namespace if available, otherwise use top-level options
                # sift_specific = getattr(sift_opts, "sift", sift_opts)
                
                sift_opts.peak_threshold = self.conf.detection_threshold
                sift_opts.edge_threshold = self.conf.edge_threshold
                sift_opts.first_octave = self.conf.first_octave
                sift_opts.num_octaves = self.conf.num_octaves
                sift_opts.normalization = pycolmap.Normalization.L2  # L1_ROOT is buggy.
                sift_opts.max_num_features = self.conf.max_num_keypoints

                device = (
                    pycolmap.Device.auto if backend == "pycolmap"
                    else getattr(pycolmap.Device, backend.replace("pycolmap_", ""))
)
            self.sift = pycolmap.Sift(options=options, device=device)
        elif backend == "opencv":
            self.sift = cv2.SIFT_create(
                contrastThreshold=self.conf.detection_threshold,
                nfeatures=self.conf.max_num_keypoints,
                edgeThreshold=self.conf.edge_threshold,
                nOctaveLayers=self.conf.num_octaves,
            )
        elif backend in {"py_cudasift", "py_Cudasift", "py_CudaSift"}:
            if cudasift_py is None:
                raise ImportError(
                    "Cannot find module cudasift_py: install/build the pybind11 wrapper "
                    "or use another backend."
                )
        else:
            backends = {
                "opencv",
                "pycolmap",
                "pycolmap_cpu",
                "pycolmap_cuda",
                "py_cudasift",
                "py_Cudasift",
                "py_CudaSift"
            }
            raise ValueError(
                f"Unknown backend: {backend} not in " f"{{{','.join(backends)}}}."
            )

    def extract_single_image(self, image: torch.Tensor):
        image_np = np.clip(image.cpu().numpy().squeeze(0), 0.0, 1.0)

        if self.conf.backend.startswith("pycolmap"):
            if version.parse(pycolmap.__version__) >= version.parse("0.5.0"):
                detections, descriptors = self.sift.extract(image_np)
                scores = None  # Scores are not exposed by COLMAP anymore.
            else:
                detections, scores, descriptors = self.sift.extract(image_np)
            keypoints = detections[:, :2]  # Keep only (x, y).
            scales, angles = detections[:, -2:].T
            if scores is not None and (
                self.conf.backend == "pycolmap_cpu" or not pycolmap.has_cuda
            ):
                # Normalize scores to non-negative; optional scale weighting is applied later.
                scores = np.abs(scores)
        elif self.conf.backend == "opencv":
            # TODO: Check if opencv keypoints are already in corner convention
            keypoints, scores, scales, angles, descriptors = run_opencv_sift(
                self.sift, (image_np * 255.0).astype(np.uint8)
            )
        elif self.conf.backend in {"py_cudasift", "py_Cudasift", "py_CudaSift"}:
            image_np = np.clip(image_np * 255.0, 0.0, 255.0)
            max_pts = (
                self.conf.max_num_keypoints
                if self.conf.filter_kpts_with_wrapper
                else 100000
            )
            keypoints, scales, angles, scores, descriptors = cudasift_py.extract(
                image_np.astype(np.float32, copy=False),
                num_octaves=self.conf.num_octaves,
                init_blur=float(self.conf.init_blur),
                thresh=self.conf.detection_threshold,
                lowest_scale=float(self.conf.first_octave),
                max_pts=max_pts,
                #edge_threshold is hardoceded at 10 inside CudaSift
            )
            keypoints = keypoints + 0.5         # Keypoints are not in corner convention in CudaSift
            angles = np.deg2rad(angles)
            scores = np.abs(scores)

        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": angles,
            "descriptors": descriptors,
        }
        if scores is not None:
            pred["keypoint_scores"] = scores

        # sometimes pycolmap returns points outside the image. We remove them
        if self.conf.backend.startswith("pycolmap"):
            is_inside = (
                pred["keypoints"] + 0.5 < np.array([image_np.shape[-2:][::-1]])
            ).all(-1)
            pred = {k: v[is_inside] for k, v in pred.items()}

        if self.conf.nms_radius is not None:
            keep = filter_dog_point(
                pred["keypoints"],
                pred["scales"],
                pred["oris"],
                image_np.shape,
                self.conf.nms_radius,
                pred.get("keypoint_scores", None),
            )
            pred = {k: v[keep] for k, v in pred.items()}

        pred = {k: torch.from_numpy(v) for k, v in pred.items()}

        # Keep only the top-k keypoints by score or scale
        # Since max_num_keypoints is a soft upper limit, we must re filter to check
        num_points = self.conf.max_num_keypoints
        if num_points is not None and len(pred["keypoints"]) > num_points:
            # Prefer keypoint scores if available; otherwise fall back to scales
            if "keypoint_scores" in pred:
                ranking_scores = pred["keypoint_scores"]
                if self.conf.filter_with_scale_weighting:
                    ranking_scores = ranking_scores * pred["scales"]
                indices = torch.topk(ranking_scores, num_points).indices
            else:
                # Use scales as a proxy for keypoint quality when scores are unavailable
                indices = torch.topk(pred["scales"], num_points).indices
            pred = {k: v[indices] for k, v in pred.items()}

        if len(pred["scales"]) > 0:
            sort_indices = torch.argsort(
                pred["scales"], descending=not self.conf.sort_scales_low_to_large
            )
            pred = {k: v[sort_indices] for k, v in pred.items()}

        # # Prints to debug to find optimal parameters for endomapper
        # num_keypoints = len(pred["keypoints"])
        # print(f"Number of keypoints: {num_keypoints}") 

        detected_keypoints = len(pred["keypoints"])
        padded_keypoints = 0

        if self.conf.force_num_keypoints:
            num_points = max(self.conf.max_num_keypoints, detected_keypoints)
            padded_keypoints = max(0, num_points - detected_keypoints)

            pred["keypoints"] = pad_to_length(
                pred["keypoints"],
                num_points,
                -2,
                mode="random_c",
                bounds=(0, min(image.shape[1:])),
            )
            pred["scales"] = pad_to_length(pred["scales"], num_points, -1, mode="zeros")
            pred["oris"] = pad_to_length(pred["oris"], num_points, -1, mode="zeros")
            pred["descriptors"] = pad_to_length(
                pred["descriptors"], num_points, -2, mode="zeros"
            )
            if pred.get("keypoint_scores", None) is not None:
                scores = pad_to_length(
                    pred["keypoint_scores"], num_points, -1, mode="zeros"
                )
                pred["keypoint_scores"] = scores
        pred["num_keypoints_detected"] = torch.tensor(detected_keypoints)
        pred["num_keypoints_padded"] = torch.tensor(padded_keypoints)
        return pred

    def _forward(self, data: dict) -> dict:
        image = data["image"]
        if image.shape[1] == 3:
            if self.conf.extractor_channel == "grayscale":
                image = rgb_to_grayscale(image)
            else:
                channel_map = {"red": 0, "green": 1, "blue": 2}
                if self.conf.extractor_channel not in channel_map:
                    raise ValueError(
                        "Unknown extractor_channel: "
                        f"{self.conf.extractor_channel} not in "
                        "{grayscale,red,green,blue}."
                    )
                channel_idx = channel_map[self.conf.extractor_channel]
                image = image[:, channel_idx : channel_idx + 1, ...]
        device = image.device
        image = image.cpu()
        pred = []
        for k in range(len(image)):
            img = image[k]
            if "image_size" in data.keys():
                # avoid extracting points in padded areas
                w, h = data["image_size"][k]
                w, h = int(w), int(h)
                img = img[:, :h, :w]
            p = self.extract_single_image(img)
            pred.append(p)
        pred = {k: torch.stack([p[k] for p in pred], 0).to(device) for k in pred[0]}
        if self.conf.rootsift:
            pred["descriptors"] = sift_to_rootsift(pred["descriptors"])
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
