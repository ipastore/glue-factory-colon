import logging
from pathlib import Path

import torch
from lightglue.lightglue import LightGlue  # adjust import as needed


repo_root = Path(__file__).resolve().parents[1]
log_path = Path("/home/student/glue-factory-colon/tools/convert_weights/output.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -------------------------------
# Step 1: Load your .pth and export to .pt
# -------------------------------
# your_pt = "/media/student/HDD/nacho/glue-factory/data/training_outputs/02-py_colmap+lg_MD_3D/02-py_colmap+lg_MD_3D.pt"
# your_pt = "/media/student/HDD/nacho/glue-factory/data/training_outputs/04-py_cudasift+lg_MD_3D/04-py_cudasift+lg_MD_3D.pt"
your_pt= "/media/student/HDD/nacho/glue-factory/data/training_outputs/sift_lightglue/sift_lightglue.pth"

# Load the .pth
model = LightGlue(features="sift")
torch.save(dict(model.state_dict()), your_pt)
obj_ours = torch.load(your_pt, map_location="cpu")

# -------------------------------
# Step 2: Load both the authors’ .pt and ours
# -------------------------------
authors_pt = "/media/student/HDD/nacho/glue-factory/data/training_outputs/sift_lightglue/sift_lightglue.pt"  # original author .pt aliked_lightglue.pt
obj_authors = torch.load(authors_pt, map_location="cpu")

# -------------------------------
# Step 3: Compare the two models
# -------------------------------
for key in obj_authors.keys():
    if key in obj_ours:
        logger.info(f"OK: {key} exists in both models")
        # Additionally compare the shapes of tensors
        shape_authors = obj_authors[key].shape if hasattr(obj_authors[key], "shape") else None
        shape_ours = obj_ours[key].shape if hasattr(obj_ours[key], "shape") else None
        if shape_authors == shape_ours:
            logger.info(f"Shape OK: {shape_authors}")
        else:
            logger.warning(f"Shape mismatch! Author: {shape_authors}, Ours: {shape_ours}")
    else:
        logger.warning(f": {key} does not exist in our model")


# -------------------------------
# Step 4: Compare the other direction
# -------------------------------
logger.info("\nChecking for keys in our model that are missing in the authors' model...")
for key in obj_ours.keys():
    if key in obj_authors:
        logger.info(f"OK: {key} exists in both models")
        # Additionally compare the shapes of tensors
        shape_authors = obj_authors[key].shape if hasattr(obj_authors[key], "shape") else None
        shape_ours = obj_ours[key].shape if hasattr(obj_ours[key], "shape") else None
        if shape_authors == shape_ours:
            logger.info(f"Shape OK: {shape_authors}")
        else:
            logger.warning(f"Shape mismatch! Author: {shape_authors}, Ours: {shape_ours}")
    else:
        logger.warning(f": {key} does not exist in our model")
