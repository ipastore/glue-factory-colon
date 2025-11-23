"""Check for corrupted images in the dataset."""
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Try to import OpenCV; optional but useful
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

def check_image(path: Path):
    """Check if an image can be opened using cv2.imread (same as read_image)."""
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"

    if cv2 is None:
        return False, "OpenCV (cv2) not available in this environment"

    try:
        # mirror read_image: read in color by default
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return False, "cv2.imread returned None (could not decode image)"
        # success
        return True
    except Exception as e:
        return False, f"cv2.imread exception: {e}"
    
def main():
    data_dir = Path("/media/student/HDD/nacho/glue-factory/data/endomapper/all_colmap_sequences")
    
    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\nChecking {split} split...")
        images = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))
        
        corrupted = []
        for img_path in tqdm(images):
            result = check_image(img_path)
            if result is not True:
                corrupted.append((img_path, result[1]))
                print(f"Corrupted: {img_path.name} - {result[1]}")
        
        print(f"\nFound {len(corrupted)} corrupted images in {split}")
        
        # Save list
        if corrupted:
            output = data_dir / f"corrupted_{split}_2.txt"
            with open(output, 'w') as f:
                for path, error in corrupted:
                    f.write(f"{path.name}\t{error}\n")
            print(f"Saved to {output}")

if __name__ == "__main__":
    main()