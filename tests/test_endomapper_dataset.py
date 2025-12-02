import numpy as np
import torch

from gluefactory.datasets.endomapper import Endomapper
from gluefactory.settings import DATA_PATH

def _write_endomapper_npz(tmp_path):
    data_root = DATA_PATH
    npz_root = data_root / "Endomapper_CUDASIFT_NOV25/processed_npz"
    npz_root.mkdir(parents=True, exist_ok=True)

    seq_map = "Seq_000_a_map0"
    image_names = np.array(["0", "1"], dtype=object)
    poses = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    poses[1, :3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    intrinsics = np.stack(
        [
            np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]]),
            np.array([[95.0, 0.0, 64.0], [0.0, 95.0, 64.0], [0.0, 0.0, 1.0]]),
        ],
        axis=0,
    ).astype(np.float32)
    distortion_coeffs = np.zeros((2, 4), dtype=np.float32)
    overlap_matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)

    keypoints_per_image = np.array(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[0.5, 0.5]], dtype=np.float32),
        ],
        dtype=object,
    )
    descriptors_per_image = np.array(
        [
            np.stack(
                [
                    np.ones(4, dtype=np.float32),
                    np.zeros(4, dtype=np.float32),
                ],
                axis=0,
            ),
            np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        ],
        dtype=object,
    )
    depths_per_image = np.array(
        [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.5], dtype=np.float32),
        ],
        dtype=object,
    )
    scales_per_image = np.array(
        [np.array([1.0, 1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)],
        dtype=object,
    )
    orientations_per_image = np.array(
        [np.array([0.0, 0.1], dtype=np.float32), np.array([0.2], dtype=np.float32)],
        dtype=object,
    )
    scores_per_image = np.array(
        [np.array([0.3, 0.2], dtype=np.float32), np.array([0.4], dtype=np.float32)],
        dtype=object,
    )
    point3d_ids_per_image = np.array(
        [np.array([1, 2], dtype=np.int64), np.array([2], dtype=np.int64)],
        dtype=object,
    )
    valid_depth_mask_per_image = np.array(
        [np.array([True, False]), np.array([True])],
        dtype=object,
    )
    valid_3d_mask_per_image = np.array(
        [np.array([True, True]), np.array([True])],
        dtype=object,
    )
    point3d_ids = np.array([1, 2], dtype=np.int64)
    point3d_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    out_path = npz_root / f"{seq_map}.npz"
    np.savez(
        out_path,
        image_names=image_names,
        poses=poses,
        intrinsics=intrinsics,
        distortion_coeffs=distortion_coeffs,
        overlap_matrix=overlap_matrix,
        keypoints_per_image=keypoints_per_image,
        descriptors_per_image=descriptors_per_image,
        depths_per_image=depths_per_image,
        scales_per_image=scales_per_image,
        orientations_per_image=orientations_per_image,
        scores_per_image=scores_per_image,
        point3D_ids_per_image=point3d_ids_per_image,
        valid_depth_mask_per_image=valid_depth_mask_per_image,
        valid_3d_mask_per_image=valid_3d_mask_per_image,
        point3D_ids=point3d_ids,
        point3D_coords=point3d_coords,
        map_id="0",
        seq="Seq_000_a",
    )
    return seq_map, data_root


def test_endomapper_pair_loader(tmp_path):
    seq_map, data_root = _write_endomapper_npz(tmp_path)
    max_feat = 4
    conf = {
        "data_dir": "Endomapper_CUDASIFT_NOV25/",
        "npz_subpath": "processed_npz",
        "train_split": [seq_map],
        "train_num_per_seq": None,
        "max_num_features": max_feat,
        "sort_by_overlap": True,
    }
    dataset = Endomapper(conf).get_dataset("train")

    assert len(dataset) >= 1
    sample = dataset[0]

    assert sample["overlap_0to1"] == 0.5

    view0 = sample["view0"]
    cache0 = view0["cache"]
    assert cache0["keypoints"].shape[0] == max_feat
    assert torch.allclose(
        cache0["keypoints"][:2], torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert torch.equal(
        cache0["valid_depth_keypoints"][2:], torch.zeros(max_feat - 2, dtype=torch.bool)
    )
    assert torch.allclose(
        view0["camera"].calibration_matrix(),
        torch.tensor(
            [[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )

    identity = sample["T_0to1"] @ sample["T_1to0"]
    assert torch.allclose(identity.R, torch.eye(3), atol=1e-6)
    assert torch.allclose(identity.t, torch.zeros(3), atol=1e-6)
