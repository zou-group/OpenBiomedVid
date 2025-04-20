#!/bin/bash

# Define paths and parameters
JSONL_PATH="/home/rahulthapa/repos/OpenBiomedVid/src/openbiomedvid/train/data/openbiomedvid_processed.jsonl"
BASE_PATH="/data/rahulthapa/OpenBiomedVid_test/video_segments_processed"
OUTPUT_DIR="/home/rahulthapa/repos/OpenBiomedVid/src/openbiomedvid/train/data/openbiomedvid_processed"
NUM_PROC=64
SPLIT="train"

# Run the dataset preparation script
python 2_create_training_dataset.py \
    --jsonl_path "$JSONL_PATH" \
    --base_path "$BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_proc "$NUM_PROC" \
    --split "$SPLIT"
