VIDEO_DIR="/path/to/videos"
AUDIO_DIR="/path/to/audio"
OUTPUT_DIR="/path/to/split_outputs"

mkdir -p "$OUTPUT_DIR"

python video_splitting.py \
    --video_dir "$VIDEO_DIR" \
    --audio_dir "$AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR"