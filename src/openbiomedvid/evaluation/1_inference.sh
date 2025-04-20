
echo "Current working directory: $(pwd)"

export NUM_GPUS=8
export MASTER_PORT=8848

temperature=0.0
task_name="SurgeryVideoQA"
task_video_path="/data/rahulthapa/biomed_yt_videos_processed/benchmarks/surgeryqa"

# task_name="MIMICEchoQA"
# task_video_path="/data/rahulthapa/vlm_data/physionet.org/files"

model_path="Qwen/Qwen2-VL-7B-Instruct"
output_path="./logs/Qwen2-VL-7B-Instruct"

echo "Running inference for base model on task: ${task_name}"
torchrun --nproc_per_node=${NUM_GPUS} \
    1_inference.py \
    --model_path $model_path \
    --task_name $task_name \
    --task_video_path $task_video_path \
    --output_path $output_path \
    --temperature $temperature
