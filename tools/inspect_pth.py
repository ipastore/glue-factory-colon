import torch

# Path to your weights
# file_path = "outputs/training/sift+lg_pretrained/sift_lightglue.pth" #no confidence layer, 253
# file_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin_bis.pth" # saved with the issue 118 of gluefactory: 254 without prefix
# file_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin.pth"  # saved with the issue  48 of gluefactory: 254 with prefix
# file_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/sift_lightglue1.pt"      #used in the SLAM, 254 transformers.prefix
file_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin.pt" # 254 transformers.prefix
file_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin_generic_dict.pt" #254 prefix, genericdict

# Load the file
# map_location='cpu' ensures you can open it even if you don't have a GPU
data = torch.load(file_path, map_location='cpu')

print(f"--- Inspecting: {file_path} ---")

if isinstance(data, dict):
    print(f"Total entries: {len(data)}")
    print("-" * 30)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Convert list to string first, then format
            shape_str = str(list(value.shape))
            print(f"Key: {key:<50} | Shape: {shape_str:<25} | Dtype: {value.dtype}")
        else:
            print(f"Key: {key:<50} | Value Type: {type(value)}")
else:
    print(f"Actual type: {type(data)}")