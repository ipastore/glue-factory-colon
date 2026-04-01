#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-/media/student/HDD/nacho/glue-factory/data/training_outputs}"
PORT="${2:-7007}"

echo "TRAIN_PATH: ${TRAIN_PATH}"
echo "PORT: ${PORT}"

docker run \
  -d \
  -p "${PORT}:${PORT}" \
  -v "${TRAIN_PATH}:/workspace/outputs/training" \
  --name tensorboard \
  tensorflow/tensorflow:2.13.0 \
  tensorboard   --port "${PORT}" --logdir=/workspace/outputs/training/ --host=0.0.0.0