# Quick script to check your dataset statistics
import numpy as np
from pathlib import Path

npz_dir = Path("/home/student/glue-factory-colon/data/Endomapper_CUDASIFT_DIC25/processed_npz")

print("Map Size Distribution:")
print("=" * 60)

sizes = []
for npz_file in sorted(npz_dir.glob("*.npz")):
    try:
        data = np.load(str(npz_file), allow_pickle=True)
        num_images = data["image_names"].shape[0]
        sizes.append((npz_file.stem, num_images))
        print(f"{npz_file.stem:30s}: {num_images:3d} images")
    except Exception as e:
        print(f"{npz_file.stem:30s}: ERROR - {e}")

if sizes:
    sizes_only = [s[1] for s in sizes]
    print("\n" + "=" * 60)
    print(f"Total maps: {len(sizes)}")
    print(f"Min images: {min(sizes_only)}")
    print(f"Max images: {max(sizes_only)}")
    print(f"Mean images: {np.mean(sizes_only):.1f}")
    print(f"Median images: {np.median(sizes_only):.1f}")