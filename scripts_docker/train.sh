#!/usr/bin/env bash
set -euo pipefail

# TAG="py_colmap+lg_HM_baseline"
TAG="py_colmap+lg_HM_baseline_scale_ori_flash_fp16"

# CONFIG_FILE="gluefactory/configs/sift+lightglue_homography.yaml"
CONFIG_FILE="gluefactory/configs/sift+lightglue_megadepth.yaml"

DATA_PATH="/home/ecs/glue-factory/data/"
OUTPUT_PATH="/home/ecs/glue-factory-colon/outputs"
CONFIGS_PATH="/home/ecs/glue-factory-colon/gluefactory/configs"
DOCKER_IMAGE="gluefactory:cuda11.8"

docker run \
  -d \
  --rm \
  --gpus all \
  --name train \
  --shm-size=50g \
  -v "${DATA_PATH}:/workspace/data" \
  -v "${OUTPUT_PATH}:/workspace/outputs" \
  -v "${CONFIGS_PATH}:/workspace/gluefactory/configs" \
  "${DOCKER_IMAGE}" \
  /bin/bash -lc \
    "conda run --no-capture-output -n gluefactory \
      python -m gluefactory.train ${TAG} \
      --conf ${CONFIG_FILE} \
      --mixed_precision float16 \
      2>&1 | tee /workspace/outputs/${TAG}.log"

  # -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
