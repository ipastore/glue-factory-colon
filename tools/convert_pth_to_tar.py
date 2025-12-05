from pathlib import Path

import logging
import torch
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.utils.experiments import verify_checkpoint_loading

root_weights_path = Path("outputs/training")
experiment_dir = "sift+lg_pretrained"
weight_name = "sift_lightglue.pth"

weights_path = root_weights_path / experiment_dir / weight_name
if not weights_path.exists():
    raise FileNotFoundError(f"weights not found: {weights_path}")

loaded = torch.load(weights_path, map_location="cpu")
if isinstance(loaded, dict):
    keys = list(loaded.keys())
    print(f"Keys in .pth file: {len(keys)} (showing first 10): {keys[:10]}")
else:
    print("Not a dict (raw state_dict)")

if isinstance(loaded, dict) and "model" in loaded:
    weights = loaded["model"]  # Already a full checkpoint
elif isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values()):
    weights = loaded  # Raw state dict
else:
    weights = loaded


def _add_matcher_prefix(state_dict):
    """Ensure LightGlue weights are stored under matcher.* for glue-factory models."""
    keys = list(state_dict.keys())
    if any(k.startswith("matcher.") for k in keys):
        return state_dict

    def strip_module(k: str) -> str:
        return k[len("module.") :] if k.startswith("module.") else k

    return {f"matcher.{strip_module(k)}": v for k, v in state_dict.items()}


if isinstance(weights, dict) and all(isinstance(v, torch.Tensor) for v in weights.values()):
    weights = _add_matcher_prefix(weights)

output_path = root_weights_path / experiment_dir / "checkpoint_best.tar"
output_path.parent.mkdir(parents=True, exist_ok=True)

model_conf = OmegaConf.create(
    {
        "name": "two_view_pipeline",
        "extractor": {"name": None},
        "matcher": {
            "name": "matchers.lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
    }
)

checkpoint = {
    "model": weights,
    "optimizer": {},
    "lr_scheduler": {},
    "conf": {"model": OmegaConf.to_container(model_conf, resolve=True)},
    "epoch": 0,
    "eval": {},
}

model = get_model(model_conf.name)(model_conf).eval()
verify_result = verify_checkpoint_loading(
    checkpoint, model, logger=logging.getLogger(__name__), module_prefix="matcher"
)
print(
    "Checkpoint verify summary:",
    {
        "matched": verify_result["matched"],
        "total": verify_result["total"],
        "missing": verify_result["missing"],
        "extra": verify_result["extra"],
    },
)

# Save as .tar
torch.save(checkpoint, output_path)
print(f"Checkpoint saved to {output_path}")
