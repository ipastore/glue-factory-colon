#!/bin/bash

# CONFIG_FILE="eval_py_cudasift+lightglue-finetuned"
CONFIG_FILE="eval_pycolmap+lightglue-finetuned"
# CONFIG_FILE="eval_py_cudasift+lightglue-official"
# CONFIG_FILE="eval_pycolmap+lightglue-official"
# CONFIG_FILE="roma"
# CONFIG_FILE="superpoint+superglue-official"

# CONDA_ENV="pybind11-glue" #colmap 0.6.0
# CONDA_ENV="pybind11-roma" #roma
# CONDA_ENV="gluefactory" #colmap 0.4.0


TAG="py_colmap_scores-lg_poselib_HM"

# source /home/student/anaconda3/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV}

PYTHONPATH=./ThirdParty/CudaSift-py-wrapper/build/  python -m gluefactory.eval.megadepth1500 --conf ${CONFIG_FILE} --tag ${TAG}