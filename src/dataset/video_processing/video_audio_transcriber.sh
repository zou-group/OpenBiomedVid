#!/bin/bash

VIDEO_DIR="/path/to/videos"
FRAME_OUTPUT_DIR="/path/to/frame_outputs"
AUDIO_OUTPUT_DIR="/path/to/audio_outputs"

mkdir -p "$AUDIO_OUTPUT_DIR"

python video_audio_transcriber.py \
    --video_dir "$VIDEO_DIR" \
    --frame_output_dir "$FRAME_OUTPUT_DIR" \
    --audio_output_dir "$AUDIO_OUTPUT_DIR"
