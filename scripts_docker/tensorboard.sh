#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-/home/server/glue-factory/outputs/training}"
PORT="${2:-8008}"

echo "TRAIN_PATH: ${TRAIN_PATH}"
echo "PORT: ${PORT}"

docker run \
  -d \
  -p "${PORT}:${PORT}" \
  -v "${TRAIN_PATH}:/workspace/outputs/training" \
  --name tensorboard \
  tensorflow/tensorflow:latest \
  tensorboard --logdir=/workspace/outputs/training/ --host=0.0.0.0