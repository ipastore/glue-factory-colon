import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a megadepth1500-style views.txt from Endomapper Dense scene_info NPZ files."
    )
    parser.add_argument(
        "--scene_info_dir",
        type=Path,
        default=Path("/media/student/HDD/nacho/glue-factory/data/endomapper_dense/scene_info"),
        help="Directory containing Endomapper Dense scene_info/*.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/media/student/HDD/nacho/glue-factory/data/endomapper_dense1500/views.txt"),
        help="Output views.txt path.",
    )
    parser.add_argument(
        "--seq_maps_file",
        type=Path,
        default=None,
        help="Optional text file listing seq_map names to export, one per line.",
    )
    parser.add_argument(
        "--seq_maps",
        nargs="*",
        default=None,
        help="Optional explicit seq_map names to export.",
    )
    parser.add_argument(
        "--pairs_file",
        type=Path,
        default=None,
        help="Optional pairs.txt file. If provided, only views referenced by these pairs are exported.",
    )
    return parser.parse_args()


def load_seq_maps(args):
    if args.seq_maps:
        return list(args.seq_maps)
    if args.seq_maps_file is not None:
        return [line.strip() for line in args.seq_maps_file.read_text().splitlines() if line.strip()]
    return [path.stem for path in sorted(args.scene_info_dir.glob("*.npz"))]


def format_pose(T):
    R = T[:3, :3].reshape(-1)
    t = T[:3, 3]
    return [f"{float(x):.8g}" for x in np.concatenate([R, t])]


def format_camera_params(camera, K):
    model = str(camera["model"])
    width = int(camera["width"])
    height = int(camera["height"])
    params = np.asarray(camera["params"], dtype=np.float32).copy()

    if params.size >= 4:
        params[0] = float(K[0, 0])
        params[1] = float(K[1, 1])
        params[2] = float(K[0, 2])
        params[3] = float(K[1, 2])

    return [model, str(width), str(height)] + [f"{float(x):.8g}" for x in params]


def load_used_images(pairs_file):
    used = set()
    for line in pairs_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name0, name1 = line.split()
        used.add(name0)
        used.add(name1)
    return used


def iter_view_lines(scene_info_path, used_images=None):
    seq_map = scene_info_path.stem
    with np.load(str(scene_info_path), allow_pickle=True) as data:
        image_names = data["image_names"]
        poses = data["poses"].astype(np.float32, copy=False)
        intrinsics = data["intrinsics"].astype(np.float32, copy=False)
        cameras = data["cameras"]
        camera_indices = data["camera_indices"].astype(np.int64, copy=False)

        for idx, image_name in enumerate(image_names):
            image_name = str(image_name)
            rel_path = f"{seq_map}/{image_name}"
            if used_images is not None and rel_path not in used_images:
                continue
            camera = cameras[int(camera_indices[idx])]
            fields = [rel_path]
            fields.extend(format_pose(poses[idx]))
            fields.extend(format_camera_params(camera, intrinsics[idx]))
            yield " ".join(fields)


def main():
    args = parse_args()
    seq_maps = load_seq_maps(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    used_images = load_used_images(args.pairs_file) if args.pairs_file else None

    lines = []
    for seq_map in seq_maps:
        scene_info_path = args.scene_info_dir / f"{seq_map}.npz"
        if not scene_info_path.exists():
            raise FileNotFoundError(f"scene_info file not found: {scene_info_path}")
        lines.extend(iter_view_lines(scene_info_path, used_images=used_images))

    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} views to {args.output}")


if __name__ == "__main__":
    main()
