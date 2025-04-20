#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONFIG_FILE=./accelerate_configs/deepspeed_zero2_8gpu.yaml

accelerate launch \
      --config_file ${CONFIG_FILE} \
      --main_process_ip localhost \
      --main_process_port 29500 \
      4_train_video.py \
      configs/qwen2_vl_video.yaml