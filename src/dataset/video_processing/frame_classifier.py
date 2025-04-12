import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import numpy as np
from PIL import Image
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from transformers import AutoImageProcessor, SiglipForImageClassification

import torch
import torch.multiprocessing as mp

from utils import get_video_list


def get_video_duration(video_file):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
               '-show_entries', 'stream=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
               video_file]
        output = subprocess.run(cmd, capture_output=True, text=True)
        if output.returncode == 0 and output.stdout.strip():
            duration = float(output.stdout)
            return duration
        raise RuntimeError(f"Failed to get duration for {video_file}: ffprobe returned code {output.returncode}")
    except Exception as e:
        raise RuntimeError(f"Error getting duration for {video_file}: {str(e)}")


def get_frame(video_file, timestamp, max_retries=5):
    """Simplified frame extraction with basic retry"""
    category = os.path.basename(os.path.dirname(video_file))
    filename = os.path.basename(video_file).split('.mp4')[0]
    image_path = os.path.join(IMAGE_OUTPUT_DIR, category, f"{filename}_{timestamp}.jpg")

    # Create directory if needed
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    cmd = [
        'ffmpeg', '-ss', str(timestamp), 
        '-i', video_file,
        '-vframes', '1',
        '-q:v', '2',
        '-y',
        '-loglevel', 'error',
        image_path
    ]
    
    # Simple retry loop
    for _ in range(max_retries):
        try:
            if not os.path.exists(image_path):
                subprocess.run(cmd, capture_output=True, check=True)
            return Image.open(image_path)
        except Exception:
            if os.path.exists(image_path):
                os.remove(image_path)  # Clean up failed attempt
            pass
    
    raise RuntimeError(f"Failed to extract frame at {timestamp}s")
    
def process_video(args):
    """Process a single video. Helper function for threading."""
    video_file, output_path, model, processor, device = args
    try:
        video = VideoFileClip(video_file)
        timestamps = np.arange(0, video.duration, 0.5)
        predictions_array = np.zeros((len(timestamps), 2), dtype=np.float32)
        predictions_array[:, 0] = timestamps

        for i in range(0, len(timestamps), FRAME_BATCH_SIZE):
            batch_timestamps = timestamps[i:i+FRAME_BATCH_SIZE]
            batch_frames = [video.get_frame(t) for t in batch_timestamps]
            
            with torch.no_grad():
                inputs = processor(
                    images=batch_frames,
                    return_tensors="pt",
                ).to(device)
            
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predictions_array[i:i+len(batch_timestamps), 1] = probabilities[:, 1].cpu().numpy()
            
        video.close()
        np.save(output_path, predictions_array)
        
    except Exception as e:
        #print(f"Error processing {video_file}: {str(e)}")
        if 'video' in locals():
            video.close()

def process_videos(video_paths, gpu_id):
    """Process a list of videos on a specific GPU using multiple threads."""
    device = f"cuda:{gpu_id}"
    model = SiglipForImageClassification.from_pretrained(IMAGE_CLASSIFIER_CHECKPOINT).to(device)
    processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    # Process videos using thread pool with tqdm
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_video, (video_file, output_path, model, processor, device))
            for video_file, output_path in video_paths
        ]
        
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(video_paths),
            desc=f"GPU {gpu_id}",
            position=gpu_id
        ):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process videos for frame classification')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save frame outputs')
    parser.add_argument('--image_output_dir', type=str, required=True,
                        help='Directory to save image outputs')
    parser.add_argument('--frame_batch_size', type=int, required=True,
                        help='Batch size for processing frames')
    parser.add_argument('--image_classifier_checkpoint', type=str, required=True,
                        help='Path to the fine-tuned model checkpoint')
    args = parser.parse_args()
    
    folder = args.video_dir
    output_dir = args.output_dir
    IMAGE_OUTPUT_DIR = args.image_output_dir
    FRAME_BATCH_SIZE = args.frame_batch_size
    IMAGE_CLASSIFIER_CHECKPOINT = args.image_classifier_checkpoint
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect video paths
    processed_count = 0
    video_paths = []

    video_paths = get_video_list(folder)

    for video_path in video_paths:
        category = os.path.basename(os.path.dirname(video_path))
        base_name = os.path.basename(video_path).split('.mp4')[0]
        
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, category, f"{base_name}.npy")
        
        if os.path.exists(output_path):
            try:
                predictions = np.load(output_path)
                if len(predictions) > 0:
                    processed_count += 1
                    continue
            except (EOFError, ValueError):
                pass
        video_paths.append((video_path, output_path))

    np.random.shuffle(video_paths)
    print(f"Processed {processed_count} videos")
    print(f"Total videos: {len(video_paths)}")


    # Shuffle and split videos evenly across GPUs
    num_gpus = torch.cuda.device_count()
    videos_per_gpu = len(video_paths) // num_gpus
    gpu_paths = []
    
    # Split into contiguous chunks per GPU
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if gpu_id < num_gpus - 1 else len(video_paths)
        gpu_paths.append(video_paths[start_idx:end_idx])

    # Create one process per GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_videos,
            args=(gpu_paths[gpu_id], gpu_id)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()



    

