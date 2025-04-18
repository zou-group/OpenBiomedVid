#!/bin/bash

NPY_DIR="/path/to/npy_files"
AUDIO_DIR="/path/to/audio"
VIDEO_DIR="/path/to/video"
OUTPUT_DIR="/path/to/outputs"

JSON_FILE="medical_segments.jsonl"
CONF_THRESHOLD=0.64
SPLIT="all"  # Can be 'first', 'second', or 'third'
DEBUG=0

python medical_segments.py \
    --npy_dir "$NPY_DIR" \
    --audio_dir "$AUDIO_DIR" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --json_file "$JSON_FILE" \
    --confidence_threshold "$CONF_THRESHOLD" \
    --debug "$DEBUG" \
    --split "$SPLIT" 