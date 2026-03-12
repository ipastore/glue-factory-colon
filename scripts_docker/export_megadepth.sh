#!/bin/bash
set -euo pipefail

METHOD="pycolmap-sift-gpu"

DATA_PATH="/home/ecs/glue-factory/data/"
OUTPUT_PATH="/home/ecs/glue-factory-colon/outputs"
CONFIGS_PATH="/home/ecs/glue-factory-colon/gluefactory/configs"
DOCKER_IMAGE="official-gluefactory:cuda11.8"

docker run -d \
  --rm \
  --gpus all \
  --shm-size=50g \
  -v "${DATA_PATH}:/workspace/data" \
  -v "${OUTPUT_PATH}:/workspace/outputs" \
  -v "${CONFIGS_PATH}:/workspace/gluefactory/configs" \
  "${DOCKER_IMAGE}" \
  /bin/bash -lc \
    "conda run --no-capture-output -n gluefactory \
      python -m gluefactory.scripts.export_megadepth \
      --method ${METHOD}\
      --num_workers 8 \
      2>&1 | tee /workspace/outputs/export_megadepth.log"


