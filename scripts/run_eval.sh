#!/bin/bash
#
# Runs N-way K-shot evaluation on the test set.
#

echo "--- Starting Few-Shot Evaluation ---"
python -m src.pipeline.train_fewshot \
    --config configs/fewshot.yaml \
    --paths configs/paths.yaml \
    --eval_only \
    --device cuda

echo "--- Evaluation Finished ---"

