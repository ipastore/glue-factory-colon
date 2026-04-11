#!/bin/bash
set -euo pipefail

METHOD="py-cudasift"

DATA_PATH="/media/student/HDD/nacho/glue-factory/data"
OUTPUT_PATH="/home/student/glue-factory-colon/outputs"
CONFIGS_PATH="/home/student/glue-factory-colon/gluefactory/configs"
DOCKER_IMAGE="gluefactory:cuda11.8"

docker run -d \
  --rm \
  --name export \
  --gpus all \
  --shm-size=50g \
  -v "${DATA_PATH}:/workspace/data" \
  -v "${OUTPUT_PATH}:/workspace/outputs" \
  -v "${CONFIGS_PATH}:/workspace/gluefactory/configs" \
  "${DOCKER_IMAGE}" \
  /bin/bash -lc \
    "conda run --no-capture-output -n gluefactory \
      python -m gluefactory.scripts.export_endomapper_roma \
      --method ${METHOD}\
      --num_workers 8 \
      2>&1 | tee /workspace/outputs/export_endomapper_roma.log"
