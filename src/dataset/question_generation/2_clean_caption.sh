#!/bin/bash

INPUT_PATH="path/to/input/file.jsonl"
OUTPUT_PATH="path/to/output/file.jsonl"

echo "Starting caption cleaning..."
echo "Input: $INPUT_PATH"

python 2_clean_caption.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_workers 32

echo "Done! Output saved as $OUTPUT_PATH"