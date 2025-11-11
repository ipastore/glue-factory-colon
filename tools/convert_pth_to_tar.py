from pathlib import Path

import torch

root_weights_path = Path("outputs/training")
experiment_dir = "sp+lg_pretrained"
weight_name = "pretrained_superpoint_lightglue.pth"

weights_path = root_weights_path / experiment_dir / weight_name
if not weights_path.exists():
    raise FileNotFoundError(f"weights not found: {weights_path}")

# First, inspect what's in the .pth file
loaded = torch.load(weights_path, map_location="cpu")
if isinstance(loaded, dict):
    print("Keys in .pth file:", loaded.keys())
else:
    print("Not a dict (raw state_dict)")
# print("Content preview:", {k: type(v) for k, v in loaded.items()}
#       if isinstance(loaded, dict) else "State dict tensors")

# If it's a raw state_dict (tensors), use it directly
if isinstance(loaded, dict) and "model" in loaded:
    weights = loaded["model"]  # Already a full checkpoint
elif isinstance(loaded, dict) and all(
    isinstance(v, torch.Tensor) for v in loaded.values()
):
    weights = loaded  # Raw state dict
else:
    weights = loaded

output_path = root_weights_path / experiment_dir / "checkpoint_best.tar"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Create a checkpoint with minimal config
checkpoint = {
    "model": weights,
    "optimizer": {},
    "lr_scheduler": {},
    "conf": {
        "model": {
            "name": "superpoint_lightglue",  # adjust to your model name
        }
    },
    "epoch": 0,
    "eval": {},
}

# Save as .tar
torch.save(checkpoint, output_path)
print(f"Checkpoint saved to {output_path}")
