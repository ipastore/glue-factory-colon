"""Check for corrupted/missing NPZ files and their referenced images in Endomapper."""
import argparse
import sys
from pathlib import Path
import numpy as np
import zipfile
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

# Expected keys in Endomapper NPZ files based on the preprocessing script.
EXPECTED_KEYS = {
    "image_names",
    "image_sizes",
    "camera_ids",
    "cameras",
    "camera_indices",
    "poses",
    "intrinsics",
    "map_id",
    "seq",
    "overlap_matrix",
    "keypoints_per_image",
    "descriptors_per_image",
    "depths_per_image",
    "scales_per_image",
    "orientations_per_image",
    "scores_per_image",
    "point3D_ids_per_image",
    "valid_depth_mask_per_image",
    "valid_3d_mask_per_image",
    "point3D_ids",
    "point3D_coords",
}

def check_image(path: Path):
    """Check if an image can be opened using cv2.imread (same as read_image in gluefactory)."""
    if not path.exists():
        return False, f"File not found: {path}"

    if cv2 is None:
        return False, "OpenCV (cv2) not available"

    try:
        # Mirror read_image: read in color by default
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return False, "cv2.imread returned None (could not decode image)"
        # Verify we can access the array
        _ = image.shape
        return True, None
    except Exception as e:
        return False, f"cv2.imread exception: {e}"

def check_npz(
    path: Path,
    data_root: Path,
    check_images: bool = True,
    full_read: bool = False,
):
    """Check if an NPZ file can be opened and loaded properly, optionally check images."""
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}", []

    try:
        # Try to load the NPZ file
        data = np.load(str(path), allow_pickle=True)
        
        # Check if we can access the keys
        keys = set(data.keys())
        
        missing_keys = EXPECTED_KEYS - keys
        if missing_keys:
            data.close()
            return False, f"Missing keys: {missing_keys}", []
        
        # Try to access each array to ensure it's not corrupted
        for key in keys:
            try:
                arr = data[key]
                # Try to get shape (will fail if corrupted)
                if hasattr(arr, 'shape'):
                    _ = arr.shape
                # For object arrays (ragged), try to access first element if exists
                if arr.dtype == object and len(arr) > 0:
                    _ = arr[0]
                if full_read:
                    if arr.dtype == object:
                        for i in range(len(arr)):
                            _ = arr[i]
                            if hasattr(_, "shape"):
                                _ = _.shape
                    else:
                        _ = np.asarray(arr)
            except Exception as e:
                data.close()
                zip_info = None
                try:
                    with zipfile.ZipFile(path) as zf:
                        bad_member = zf.testzip()
                    if bad_member:
                        zip_info = f"zipfile.testzip bad member: {bad_member}"
                    else:
                        zip_info = "zipfile.testzip did not find bad members"
                except Exception as zip_error:
                    zip_info = f"zipfile.testzip failed: {zip_error}"
                return False, f"Error accessing key '{key}': {e} | {zip_info}", []
        
        # Check referenced images if requested
        missing_images = []
        if check_images:
            try:
                seq_name = str(np.asarray(data['seq']).item())
                map_id = str(np.asarray(data['map_id']).item())
                image_names = data['image_names']
                
                # Build path to keyframes directory
                keyframes_dir = data_root / seq_name / "output" / "3D_maps" / map_id / "keyframes"
                
                if not keyframes_dir.exists():
                    data.close()
                    return False, f"Keyframes directory not found: {keyframes_dir}", []
                
                for img_name in image_names:
                    img_name_str = str(np.asarray(img_name).item())
                    keyframe_path = keyframes_dir / f"Keyframe_{img_name_str}.png"
                    
                    result, error = check_image(keyframe_path)
                    if not result:
                        missing_images.append((keyframe_path, error))
                        
            except Exception as e:
                data.close()
                zip_info = None
                try:
                    with zipfile.ZipFile(path) as zf:
                        bad_member = zf.testzip()
                    if bad_member:
                        zip_info = f"zipfile.testzip bad member: {bad_member}"
                    else:
                        zip_info = "zipfile.testzip did not find bad members"
                except Exception as zip_error:
                    zip_info = f"zipfile.testzip failed: {zip_error}"
                return False, f"Error checking images: {e} | {zip_info}", []
        
        # Success
        data.close()
        
        if missing_images:
            return False, f"NPZ valid but {len(missing_images)} images missing/corrupted", missing_images
        
        return True, None, []
        
    except Exception as e:
        zip_info = None
        try:
            with zipfile.ZipFile(path) as zf:
                bad_member = zf.testzip()
            if bad_member:
                zip_info = f"zipfile.testzip bad member: {bad_member}"
            else:
                zip_info = "zipfile.testzip did not find bad members"
        except Exception as zip_error:
            zip_info = f"zipfile.testzip failed: {zip_error}"
        return False, f"Failed to load NPZ: {e} | {zip_info}", []

def main():
    parser = argparse.ArgumentParser(
        description="Check for corrupted/missing NPZ files and referenced images."
    )
    parser.add_argument(
        "--full-read",
        action="store_true",
        help="Force full read of each NPZ array (slower, catches deeper corruption).",
    )
    args = parser.parse_args()

    # Root directory containing sequences
    data_root = Path("/media/student/HDD/nacho/glue-factory/data/slam-results-nacho")
    npz_subdir = "processed_npz"
    
    npz_dir = data_root / npz_subdir
    
    if not npz_dir.exists():
        print(f"Error: NPZ directory not found: {npz_dir}")
        print(f"Please check the path in the script.")
        return 1
    
    if cv2 is None:
        print("Warning: OpenCV not available. Image checking will be skipped.")
        check_images = False
    else:
        check_images = True
    
    print(f"Checking NPZ files in: {npz_dir}")
    print(f"Image validation: {'ENABLED' if check_images else 'DISABLED'}")
    print(f"Full read: {'ENABLED' if args.full_read else 'DISABLED'}")
    npz_files = sorted(npz_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return 1
    
    print(f"Found {len(npz_files)} NPZ files to check\n")
    
    corrupted_npz = []
    all_missing_images = []
    
    for npz_path in tqdm(npz_files, desc="Checking NPZ files"):
        result, error, missing_images = check_npz(
            npz_path,
            data_root,
            check_images=check_images,
            full_read=args.full_read,
        )
        
        if not result:
            corrupted_npz.append((npz_path, error))
            print(f"\n✗ Problem: {npz_path.name}")
            print(f"  Error: {error}")
            
            if missing_images:
                print(f"  Missing/corrupted images: {len(missing_images)}")
                for img_path, img_error in missing_images[:3]:  # Show first 3
                    print(f"    - {img_path.name}: {img_error}")
                if len(missing_images) > 3:
                    print(f"    ... and {len(missing_images) - 3} more")
                all_missing_images.extend([(npz_path.name, img_path, img_error) 
                                          for img_path, img_error in missing_images])
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total NPZ files: {len(npz_files)}")
    print(f"  Valid NPZ files: {len(npz_files) - len(corrupted_npz)}")
    print(f"  Corrupted/Problematic NPZ: {len(corrupted_npz)}")
    if check_images:
        print(f"  Total missing/corrupted images: {len(all_missing_images)}")
    print(f"{'='*60}")
    
    # Save list of corrupted NPZ files
    if corrupted_npz:
        output = npz_dir / "corrupted_npz_files.txt"
        with open(output, 'w') as f:
            f.write("# Corrupted or problematic NPZ files\n")
            f.write(f"# Total: {len(corrupted_npz)}\n")
            f.write("#\n")
            for path, error in corrupted_npz:
                f.write(f"{path.name}\t{error}\n")
        print(f"\nSaved corrupted NPZ list to: {output}")
    
    # Save list of missing images
    if all_missing_images:
        output = npz_dir / "missing_images.txt"
        with open(output, 'w') as f:
            f.write("# Missing or corrupted images referenced by NPZ files\n")
            f.write(f"# Total: {len(all_missing_images)}\n")
            f.write("# Format: NPZ_FILE\tIMAGE_PATH\tERROR\n")
            f.write("#\n")
            for npz_name, img_path, error in all_missing_images:
                f.write(f"{npz_name}\t{img_path}\t{error}\n")
        print(f"Saved missing images list to: {output}")
    
    if corrupted_npz or all_missing_images:
        return 1
    else:
        print("\n✓ All NPZ files and their images are valid!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
