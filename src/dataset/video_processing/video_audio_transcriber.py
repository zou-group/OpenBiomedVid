import os
import argparse
import json
import random

import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils import get_video_list


def extract_audio(video_tuple):
    """Extract audio from a single video file"""
    video_file, output_path = video_tuple

    audio_path = os.path.join(
        os.path.dirname(output_path),
        f'{video_file.split("/")[-1].split(".")[0]}.mp3'
    )

    try:
        video = VideoFileClip(video_file, verbose=False)
        if video.audio is None:
            return
            
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return (audio_path, output_path)
    except Exception as e:
        print(f"Error extracting audio from {video_file}: {str(e)}")
        return

def process_videos(video_paths, gpu_id):
    """Process a list of videos on a specific GPU with internal batching."""
    # Initialize model for this process
    device = f"cuda:{gpu_id}"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        use_safetensors=True,
        attn_implementation="eager"
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)

    BATCH_SIZE = 2
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps="word",
        batch_size=BATCH_SIZE,
        generate_kwargs={
            "task": "translate",
            "language": "en",
            "use_cache": True,
        }
    )
    
    pbar = tqdm(total=len(video_paths), 
                desc=f"GPU {gpu_id}", 
                position=gpu_id,
                leave=True)
    
    for i in range(0, len(video_paths), BATCH_SIZE):
        batch_videos = video_paths[i:i + BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            audio_files = list(executor.map(extract_audio, batch_videos))

        audio_files = [audio_file for audio_file in audio_files if audio_file]
        
        if audio_files:
            audio_paths, output_paths = zip(*audio_files)
            
            try:
                results = pipe(list(audio_paths), batch_size=len(audio_paths))
            except Exception as e:
                print(f"Error processing batch: {e}")
                #Fall back to individual processing
                successful_results = []
                successful_paths = []
                for audio_path, output_path in zip(audio_paths, output_paths):
                    try:
                        result = pipe(audio_path)
                        successful_results.append(result)
                        successful_paths.append(output_path)
                    except Exception as e:
                        continue

                results = successful_results
                output_paths = successful_paths

            for result, output_path in zip(results, output_paths):
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
            
        pbar.update(len(batch_videos))
    
    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio from videos')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--frame_output_dir', type=str, required=True,
                        help='Directory containing frame outputs')
    parser.add_argument('--audio_output_dir', type=str, required=True,
                        help='Directory to save audio transcriptions')
    args = parser.parse_args()
    
    folder = args.video_dir
    frame_output_dir = args.frame_output_dir
    audio_output_dir = args.audio_output_dir

    processed_count = 0
    video_paths = []
    
    video_paths = get_video_list(folder)

    for video_path in video_paths:

        category = os.path.basename(os.path.dirname(video_path))
        base_name = os.path.basename(video_path).split('.mp4')[0]
                
        category_dir = os.path.join(audio_output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        output_path = os.path.join(audio_output_dir, category, f"{base_name}.json")
        
        if os.path.exists(output_path):
            processed_count += 1
        elif os.path.exists(os.path.join(frame_output_dir, category, f"{base_name}.npy")):
            video_paths.append((video_path, output_path))
                

    print(f"Found {processed_count} already processed files")
    print(f"Remaining videos to process: {len(video_paths)}")

    # Shuffle and split videos evenly across GPUs
    random.shuffle(video_paths)
    num_gpus = torch.cuda.device_count()
    videos_per_gpu = len(video_paths) // num_gpus
    gpu_paths = []
    
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