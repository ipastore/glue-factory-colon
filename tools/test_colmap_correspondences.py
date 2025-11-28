"""Bidirectional consistency checker for COLMAP text exports."""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

Point2D = Tuple[float, float, int]
TrackRef = Tuple[int, int]
MIN_OBS_FORWARD = 3


def _read_valid_lines(path: Path) -> List[str]:
    """Return non-empty, non-comment lines."""
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def parse_images_txt(path: Path) -> Dict[int, Dict[str, List[Point2D]]]:
    """Parse COLMAP images.txt into a dictionary keyed by IMAGE_ID."""
    lines = _read_valid_lines(path)
    images: Dict[int, Dict[str, List[Point2D]]] = {}
    if len(lines) % 2 != 0:
        raise ValueError("images.txt must contain pose and points2D lines in pairs.")

    for pose_line, points_line in zip(lines[0::2], lines[1::2]):
        pose_parts = pose_line.split()
        image_id = int(pose_parts[0])
        points_tokens = points_line.split()
        if len(points_tokens) % 3 != 0:
            raise ValueError(
                f"Image {image_id} has incomplete (x, y, point3D_id) triplets."
            )
        points2d: List[Point2D] = []
        for i in range(0, len(points_tokens), 3):
            x, y, point3d_id = (
                float(points_tokens[i]),
                float(points_tokens[i + 1]),
                int(points_tokens[i + 2]),
            )
            points2d.append((x, y, point3d_id))
        images[image_id] = {"name": pose_parts[-1], "points2D": points2d}
    return images


def parse_points3d_txt(path: Path) -> Dict[int, Dict[str, List[TrackRef]]]:
    """Parse COLMAP points3D.txt into a dictionary keyed by POINT3D_ID."""
    points3d: Dict[int, Dict[str, List[TrackRef]]] = {}
    for line in _read_valid_lines(path):
        parts = line.split()
        if len(parts) < 9:
            raise ValueError("points3D.txt lines must include a TRACK section.")
        point3d_id = int(parts[0])
        track_tokens = parts[8:]
        if len(track_tokens) % 2 != 0:
            raise ValueError(f"Point3D {point3d_id} track has dangling entries.")
        track: List[TrackRef] = []
        for i in range(0, len(track_tokens), 2):
            track.append((int(track_tokens[i]), int(track_tokens[i + 1])))
        points3d[point3d_id] = {"track": track}
    return points3d


def collect_correspondence_mismatches(
    images_dict: Dict[int, Dict[str, List[Point2D]]],
    points3d_dict: Dict[int, Dict[str, List[TrackRef]]],
) -> Tuple[List[str], Dict[str, int]]:
    """Gather all forward/backward inconsistencies and metrics."""
    mismatches: List[str] = []
    forward_total = backward_total = 0
    forward_ok = backward_ok = 0
    forward_hits = set()
    backward_hits = set()
    # Count how many times each point3D is referenced in images.txt
    point3d_obs_counts: Dict[int, int] = {}
    for image_data in images_dict.values():
        for _, _, point3d_id in image_data["points2D"]:
            if point3d_id == -1:
                continue
            point3d_obs_counts[point3d_id] = point3d_obs_counts.get(point3d_id, 0) + 1

    for image_id, image_data in images_dict.items():
        points2d = image_data["points2D"]
        for point2d_idx, (_, _, point3d_id) in enumerate(points2d):
            if point3d_id == -1:
                continue
            obs_count = point3d_obs_counts.get(point3d_id, 0)
            if obs_count < MIN_OBS_FORWARD:
                continue
            forward_total += 1
            point3d_data = points3d_dict.get(point3d_id)
            if point3d_data is None:
                mismatches.append(
                    f"Forward: image {image_id} point2D {point2d_idx} "
                    f"references missing point3D {point3d_id} "
                    f"(observed {obs_count}x in images.txt)"
                )
                continue
            track_refs = set(point3d_data["track"])
            if (image_id, point2d_idx) not in track_refs:
                mismatches.append(
                    f"Forward: image {image_id} point2D {point2d_idx} -> point3D "
                    f"{point3d_id} missing in points3D track "
                    f"(observed {obs_count}x in images.txt)"
                )
                continue
            forward_ok += 1
            forward_hits.add((point3d_id, image_id, point2d_idx))

    for point3d_id, point3d_data in points3d_dict.items():
        for image_id, point2d_idx in point3d_data["track"]:
            backward_total += 1
            image_data = images_dict.get(image_id)
            if image_data is None:
                mismatches.append(
                    f"Backward: point3D {point3d_id} track references "
                    f"missing image {image_id}"
                )
                continue
            points2d = image_data["points2D"]
            if point2d_idx >= len(points2d):
                mismatches.append(
                    f"Backward: point3D {point3d_id} track references image "
                    f"{image_id} point2D index {point2d_idx} out of range "
                    f"(len={len(points2d)})"
                )
                continue
            observed_point3d_id = points2d[point2d_idx][2]
            if observed_point3d_id != point3d_id:
                mismatches.append(
                    f"Backward: point3D {point3d_id} track references image "
                    f"{image_id} point2D {point2d_idx} but image points to "
                    f"{observed_point3d_id}"
                )
                continue
            backward_ok += 1
            backward_hits.add((point3d_id, image_id, point2d_idx))
    metrics = {
        "forward_total": forward_total,
        "forward_ok": forward_ok,
        "backward_total": backward_total,
        "backward_ok": backward_ok,
        "bidirectional": len(forward_hits & backward_hits),
    }
    return mismatches, metrics


def _write_synthetic_files(tmpdir: Path) -> Tuple[Path, Path]:
    images_file = tmpdir / "images.txt"
    points3d_file = tmpdir / "points3D.txt"

    images_file.write_text(
        "\n".join(
            [
                "# Header",
                "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                "# POINTS2D[] as (X, Y, POINT3D_ID)",
                "1 0 0 0 1 0 0 0 1 img1.jpg",
                "10.0 20.0 5 30.5 40.5 5",
                "2 0 0 0 1 0 0 0 1 img2.jpg",
                "11.0 21.0 5 31.5 41.5 6",
            ]
        )
    )
    points3d_file.write_text(
        "\n".join(
            [
                "# Header",
                "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
                "5 0 0 0 255 255 255 0.5 1 0 1 1 2 0",
                "6 1 1 1 255 255 255 1.0 2 1",
            ]
        )
    )
    return images_file, points3d_file


def run_check(images_path: Path, points3d_path: Path) -> Tuple[List[str], Dict[str, int]]:
    images_dict = parse_images_txt(images_path)
    points3d_dict = parse_points3d_txt(points3d_path)
    return collect_correspondence_mismatches(images_dict, points3d_dict)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate bidirectional consistency between COLMAP images.txt and points3D.txt."
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images.txt export from COLMAP.",
    )
    parser.add_argument(
        "--points3d",
        type=Path,
        help="Path to points3D.txt export from COLMAP.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write a detailed mismatch report.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run against a built-in synthetic example instead of provided paths.",
    )
    return parser.parse_args()


def _write_report(
    mismatches: List[str], report_path: Path, consistent: bool, metrics: Dict[str, int]
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "COLMAP correspondence consistency report",
        f"Status: {'consistent' if consistent else 'inconsistent'}",
        "",
    ]
    metrics_lines = [
        f"Forward references (>= {MIN_OBS_FORWARD} obs): "
        f"{metrics['forward_ok']}/{metrics['forward_total']} consistent "
        f"({metrics['forward_total'] - metrics['forward_ok']} mismatches)",
        f"Backward references: {metrics['backward_ok']}/{metrics['backward_total']} consistent "
        f"({metrics['backward_total'] - metrics['backward_ok']} mismatches)",
        f"Bidirectional consistent pairs: {metrics['bidirectional']}",
        "",
    ]
    if mismatches:
        counts = {
            "Forward": len([m for m in mismatches if m.startswith("Forward")]),
            "Backward": len([m for m in mismatches if m.startswith("Backward")]),
        }
        summary = [
            f"Total mismatches: {len(mismatches)}",
            f"Forward mismatches: {counts['Forward']}",
            f"Backward mismatches: {counts['Backward']}",
            "",
            "Detailed mismatches:",
        ]
        body = [f"- {msg}" for msg in mismatches]
    else:
        summary = ["No mismatches found."]
        body = []
    report_path.write_text("\n".join(header + metrics_lines + summary + body))


def main() -> int:
    args = _parse_args()

    if args.test:
        with tempfile.TemporaryDirectory() as tmp:
            images_path, points3d_path = _write_synthetic_files(Path(tmp))
            mismatches, metrics = run_check(images_path, points3d_path)
    else:
        if args.images is None or args.points3d is None:
            print("Error: --images and --points3d are required unless --test is used.")
            return 2
        mismatches, metrics = run_check(args.images, args.points3d)

    if mismatches:
        print(
            f"Forward (>= {MIN_OBS_FORWARD} obs): {metrics['forward_ok']}/"
            f"{metrics['forward_total']} consistent "
            f"({metrics['forward_total'] - metrics['forward_ok']} mismatches)"
        )
        print(
            f"Backward: {metrics['backward_ok']}/{metrics['backward_total']} consistent "
            f"({metrics['backward_total'] - metrics['backward_ok']} mismatches)"
        )
        print(f"Bidirectional consistent pairs: {metrics['bidirectional']}")
        print(f"Found {len(mismatches)} mismatches:")
        for msg in mismatches:
            print(f"- {msg}")
        if args.report:
            _write_report(mismatches, args.report, consistent=False, metrics=metrics)
        return 1

    print("All correspondences are consistent.")
    print(
        f"Forward (>= {MIN_OBS_FORWARD} obs): {metrics['forward_ok']}/"
        f"{metrics['forward_total']} consistent "
        f"({metrics['forward_total'] - metrics['forward_ok']} mismatches)"
    )
    print(
        f"Backward: {metrics['backward_ok']}/{metrics['backward_total']} consistent "
        f"({metrics['backward_total'] - metrics['backward_ok']} mismatches)"
    )
    print(f"Bidirectional consistent pairs: {metrics['bidirectional']}")
    if args.report:
        _write_report([], args.report, consistent=True, metrics=metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
