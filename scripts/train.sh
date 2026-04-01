#!/bin/bash

MODULE="gluefactory.train"
TAG="take_all_pairs_endomapperDense"

# CONFIG_FILE="gluefactory/configs/sift+lightglue_homography.yaml"
# CONFIG_FILE="gluefactory/configs/sift+lightglue_megadepth.yaml"
CONFIG_FILE="gluefactory/configs/py_cudasift+lightglue_endomapper_dense.yaml"

PYTHONPATH=./ThirdParty/CudaSift-py-wrapper/build python -m ${MODULE} ${TAG} --conf ${CONFIG_FILE}