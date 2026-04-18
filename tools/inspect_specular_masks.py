import argparse
from pathlib import Path

import cv2
import numpy as np


def decode_mask(npz_path: Path) -> np.ndarray:
    with np.load(str(npz_path)) as data:
        if "mask_packbits" in data and "mask_shape" in data:
            h, w = data["mask_shape"].astype(np.int64).tolist()
            flat = np.unpackbits(data["mask_packbits"], count=int(h * w))
            mask = flat.reshape((h, w)).astype(bool, copy=False)
            return mask
        if "mask" in data:
            return data["mask"].astype(bool, copy=False)
    raise KeyError(f"Missing mask keys in {npz_path}. Expected mask_packbits+mask_shape.")


def list_npz_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if recursive:
        return sorted(input_path.rglob("*.npz"))
    return sorted(input_path.glob("*.npz"))


def out_path_for(npz_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / f"{npz_path.stem}.png"
    rel = npz_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".png")


def infer_image_path(npz_path: Path, input_root: Path, image_root: Path | None) -> Path | None:
    if input_root.is_file():
        rel = npz_path.name
    else:
        try:
            rel = npz_path.relative_to(input_root)
        except Exception:
            return None
    # Roma layout:
    # <seq>/output/3D_maps/<map_id>/specular_masks/Spec_<id>.npz
    # -> <seq>/output/3D_maps/<map_id>/keyframes/Keyframe_<id>.(png|jpg|jpeg)
    if npz_path.parent.name == "specular_masks":
        stem = npz_path.stem
        if stem.startswith("Spec_"):
            keyframe_id = stem[len("Spec_") :]
            cand_dir = npz_path.parent.parent / "keyframes"
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                p = cand_dir / f"Keyframe_{keyframe_id}{ext}"
                if p.exists():
                    return p
    if image_root is None:
        return None
    # specular_undistorted/Seq_xxx/map_id/<stem>_spec.npz
    # -> Undistorted_SfM/Seq_xxx/map_id/images/<stem>.(png|jpg|jpeg)
    stem = rel.stem
    if stem.endswith("_spec"):
        stem = stem[: -len("_spec")]
    cand_dir = image_root / rel.parent / "images"
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = cand_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode cached Endomapper specular masks to PNG previews.")
    parser.add_argument("--input", type=Path, required=True, help="Input .npz file or directory containing .npz masks.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/specular_masks_preview"), help="Directory for PNG previews.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory for .npz files.")
    parser.add_argument("--mode", choices=["specular", "keep"], default="specular", help="specular: white=specular; keep: white=non-specular.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional image root (e.g. .../Undistorted_SfM) to save overlays.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of files to export.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs.")
    parser.add_argument("--overlay-alpha", type=float, default=0.65, help="Blend weight for excluded regions in overlay previews.")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    npz_files = list_npz_files(input_path, recursive=args.recursive)
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    if len(npz_files) == 0:
        print(f"No .npz files found in {input_path}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    failed = 0
    for npz_path in npz_files:
        out_path = out_path_for(npz_path, input_path, args.output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            mask_keep = decode_mask(npz_path)
            vis = (~mask_keep) if args.mode == "specular" else mask_keep
            vis_u8 = vis.astype(np.uint8, copy=False) * 255
            ok = cv2.imwrite(str(out_path), vis_u8)
            if not ok:
                raise OSError(f"cv2.imwrite failed for {out_path}")

            img_path = infer_image_path(npz_path, input_path, args.image_root)
            if img_path is not None:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None and img.shape[:2] == mask_keep.shape:
                    overlay = img.copy()
                    excluded = ~mask_keep
                    overlay[excluded] = (0, 0, 255)  # red in BGR
                    overlay = cv2.addWeighted(
                        img,
                        1.0 - args.overlay_alpha,
                        overlay,
                        args.overlay_alpha,
                        0.0,
                    )
                    overlay[excluded] = np.clip(
                        overlay[excluded].astype(np.int16) + np.array([0, 0, 40], dtype=np.int16),
                        0,
                        255,
                    ).astype(np.uint8)
                    overlay_path = out_path.with_name(out_path.stem + "_overlay.png")
                    ok_ov = cv2.imwrite(str(overlay_path), overlay)
                    if not ok_ov:
                        raise OSError(f"cv2.imwrite failed for {overlay_path}")
            exported += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] {npz_path}: {exc}")

    print(f"Done. exported={exported} skipped={skipped} failed={failed} out={args.output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
