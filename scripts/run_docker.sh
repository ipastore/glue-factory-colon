#!/bin/bash


MODULE="gluefactory.train"
TAG="py_colmap_00066667_min_scale+lg_HM_srvr"

CONFIG_FILE="gluefactory/configs/sift+lightglue_homography.yaml"
# CONFIG_FILE="gluefactory/configs/sift+lightglue_megadepth.yaml"


docker run -d --rm --gpus all \
  --shm-size=64g \
  -v /mnt/HARD/nacho/gluefactory/data:/workspace/data \
  -v /home/server/glue-factory-colon/outputs:/workspace/outputs \
  -v /home/server/glue-factory-colon/gluefactory/configs:/workspace/gluefactory/configs \
  gluefactory:cuda11.8 \
/bin/bash -lc "conda run --no-capture-output -n gluefactory python -m ${MODULE} ${TAG} --conf ${CONFIG_FILE} 2>&1 | tee /workspace/outputs/docker_stdout_stderr.log"