"""Check for corrupted images in the dataset."""
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def check_image(path):
    """Check if an image can be opened."""
    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open for actual load test
        with Image.open(path) as img:
            img.load()
        return True
    except Exception as e:
        return False, str(e)

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
            output = data_dir / f"corrupted_{split}.txt"
            with open(output, 'w') as f:
                for path, error in corrupted:
                    f.write(f"{path.name}\t{error}\n")
            print(f"Saved to {output}")

if __name__ == "__main__":
    main()