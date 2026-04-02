#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="/media/student/HDD/nacho/glue-factory/data"
OUTPUT_PATH="/home/student/glue-factory-colon/outputs"
CONFIGS_PATH="/home/student/glue-factory-colon/gluefactory/configs"
SCRIPTS_PATH="/home/student/glue-factory-colon/scripts"
DOCKER_IMAGE="gluefactory:cuda11.8"

docker run \
  -d \
  --rm \
  --gpus all \
  --name eval_hpatches \
  --shm-size=50g \
  -v /home/student/.cache/matplotlib:/home/student/.cache/matplotlib \
  -v /home/student/.cache/torch:/home/student/.cache/torch \
  -e TORCH_HOME=/home/student/.cache/torch \
  -e XDG_CACHE_HOME=/home/student/.cache \
  -v "${DATA_PATH}:/workspace/data" \
  -v "${OUTPUT_PATH}:/workspace/outputs" \
  -v "${CONFIGS_PATH}:/workspace/gluefactory/configs" \
  -v "${SCRIPTS_PATH}:/workspace/scripts" \
  "${DOCKER_IMAGE}" \
  /bin/bash -lc \
    "conda run --no-capture-output -n gluefactory \
    /workspace/scripts/eval_hpatches.sh \
    2>&1 | tee /workspace/outputs/eval_hpatches.log"


