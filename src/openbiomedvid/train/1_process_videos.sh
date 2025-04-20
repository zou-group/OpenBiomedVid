#!/bin/bash

# Set paths and parameters
BASE_PATH="/data/rahulthapa/OpenBiomedVid_test/video_segments"
DATASET="OpenBiomedVid"
OUTPUT_PATH="/data/rahulthapa/OpenBiomedVid_test/video_segments_processed"
NUM_PROCESSES=32
MAX_FRAMES=420
MAX_DURATION=420
MAX_TOKENS=16384
OUTPUT_JSONL="openbiomedvid_processed.jsonl"

# Run the script
python 1_process_videos.py \
    --base_path "$BASE_PATH" \
    --dataset "$DATASET" \
    --output_path "$OUTPUT_PATH" \
    --num_processes "$NUM_PROCESSES" \
    --max_frames "$MAX_FRAMES" \
    --max_duration "$MAX_DURATION" \
    --max_tokens "$MAX_TOKENS" \
    --output_jsonl "$OUTPUT_JSONL"
