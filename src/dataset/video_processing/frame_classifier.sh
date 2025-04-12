#!/bin/bash

VIDEO_DIR="/path/to/videos"
OUTPUT_DIR="/path/to/outputs"
IMAGE_OUTPUT_DIR="/path/to/image_outputs"
FRAME_BATCH_SIZE=256
IMAGE_CLASSIFIER_CHECKPOINT="/path/to/image_classifier_checkpoint"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$IMAGE_OUTPUT_DIR"

python src/dataset/frame_classifier.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --image_output_dir "$IMAGE_OUTPUT_DIR" \
    --frame_batch_size "$FRAME_BATCH_SIZE" \
    --image_classifier_checkpoint "$IMAGE_CLASSIFIER_CHECKPOINT"