#!/bin/bash
#
# Runs inference on a single sketch image to predict its class.
#

if [ -z "$1" ]; then
  echo "Usage: $0 path/to/sketch.png"
  exit 1
fi

IMAGE_PATH=$1

echo "--- Running Inference on ${IMAGE_PATH} ---"
python -m src.pipeline.infer \
    --config configs/fewshot.yaml \
    --paths configs/paths.yaml \
    --image "${IMAGE_PATH}" \
    --device cuda
