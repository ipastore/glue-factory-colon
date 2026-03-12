#!/bin/bash

docker run -d -p 6006:6006 \
  -v /home/server/glue-factory-colon/outputs/training:/workspace/outputs/training \
  --name tensorboard \
  tensorflow/tensorflow:latest \
  tensorboard --logdir=/workspace/outputs/training/ --host=0.0.0.0