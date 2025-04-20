#!/bin/bash

base_path="./logs/Qwen2.5-VL-7B-Instruct/logs/surgeryvideoqa"

INPUT_PATH="${base_path}/results.jsonl"
OUTPUT_PATH="${base_path}/gpt_evaluation_results.jsonl"

python3 3_gpt_eval.py \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH}


