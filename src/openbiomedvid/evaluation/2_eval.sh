#!/bin/bash

input_file="./logs/Qwen2.5-VL-7B-Instruct/logs/mimicechoqa/results.jsonl"

python3 2_eval.py \
    --input_file ${input_file}
