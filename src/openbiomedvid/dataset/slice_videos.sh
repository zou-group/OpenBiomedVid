#!/bin/bash

python slice_videos.py \
  --dataset OpenBiomedVid \
  --input_dir /data/rahulthapa/OpenBiomedVid_test/videos \
  --output_dir /data/rahulthapa/OpenBiomedVid_test/video_segments \
  --num_processes 32
