#!/bin/bash
#
# Runs the few-shot training/adaptation process.
# This script calibrates the per-class adapters on a set of few-shot episodes.
#

echo "--- Starting Few-Shot Training/Adaptation ---"
python -m src.pipeline.train_fewshot \
    --config configs/fewshot.yaml \
    --paths configs/paths.yaml \
    --device cpu

echo "--- Training/Adaptation Finished ---"

