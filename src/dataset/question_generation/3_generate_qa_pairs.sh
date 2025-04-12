#!/bin/bash

INPUT_PATH="path/to/input/file.jsonl"
OUTPUT_PATH="path/to/output/file.jsonl"

MAX_WORKERS=16

echo "Starting Q/A pair generation..."
echo "Input: $INPUT_PATH"

python 3_generate_qa_pairs.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_workers "$MAX_WORKERS"

echo "Done! Output saved as $OUTPUT_PATH"