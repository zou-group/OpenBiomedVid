import argparse
import json
import os
import math
import numpy as np
import cv2
import shutil
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
from functools import partial

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def calculate_tokens(width, height):
    width_patches = math.ceil(width / 28)
    height_patches = math.ceil(height / 28)
    return (width_patches * height_patches) // 2

def round_to_multiple_of_28(value):
    return max(round(value / 28) * 28, 56)

def calculate_new_resolution(width, height, target_tokens):
    aspect_ratio = width / height
    current_tokens = calculate_tokens(width, height)

    while current_tokens > target_tokens:
        if height * 0.95 < 56:
            width = max(round(width * 0.95), round(56 * aspect_ratio))
            height = max(round(width / aspect_ratio), 56)
        elif width * 0.95 < 56:
            height = max(round(height * 0.95), round(56 / aspect_ratio))
            width = max(round(height * aspect_ratio), 56)
        else:
            width = round(width * 0.95)
            height = round(height * 0.95)

        current_tokens = calculate_tokens(width, height)

    width_rounded = round_to_multiple_of_28(width)
    height_rounded = round_to_multiple_of_28(height)

    width_change = abs(width_rounded / width - 1)
    height_change = abs(height_rounded / height - 1)

    if width_change > height_change:
        height_rounded = round_to_multiple_of_28(width_rounded / aspect_ratio)
    else:
        width_rounded = round_to_multiple_of_28(height_rounded * aspect_ratio)

    final_tokens = calculate_tokens(width_rounded, height_rounded)
    if final_tokens > target_tokens:
        if width_rounded > height_rounded:
            width_rounded -= 28
            height_rounded = round_to_multiple_of_28(width_rounded / aspect_ratio)
        else:
            height_rounded -= 28
            width_rounded = round_to_multiple_of_28(height_rounded * aspect_ratio)

    return width_rounded, height_rounded

def process_video(entry, base_path, output_path, max_duration, max_tokens, max_frames):
    try:
        src_video_path = os.path.join(base_path, entry['video'])
        relative_path = os.path.dirname(entry['video'])
        output_dir = os.path.join(output_path, relative_path)
        ensure_dir(output_dir)

        video_filename = os.path.basename(entry['video'])
        video_name = os.path.splitext(video_filename)[0]
        processed_video_path = os.path.join(output_dir, f"{video_name}.mp4")

        cap = cv2.VideoCapture(src_video_path)
        if not cap.isOpened():
            print(f"Error opening video {entry['video']}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_duration <= max_duration:
            frame_indices = np.arange(0, total_frames, fps).astype(int)
            sampling_freq = 1
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            sampling_freq = max_frames / total_duration

        tokens_per_frame = calculate_tokens(orig_width, orig_height)
        total_tokens = tokens_per_frame * len(frame_indices)

        if total_tokens > max_tokens:
            target_tokens_per_frame = max_tokens // len(frame_indices)
            new_width, new_height = calculate_new_resolution(orig_width, orig_height, target_tokens_per_frame)
        else:
            new_width, new_height = orig_width, orig_height

        frames_extracted = 0
        temp_frame_dir = os.path.join(output_dir, f"{video_name}_temp")
        ensure_dir(temp_frame_dir)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if new_width != orig_width or new_height != orig_height:
                    frame = cv2.resize(frame, (new_width, new_height))
                frame_path = os.path.join(temp_frame_dir, f"frame_{frames_extracted:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames_extracted += 1

        cap.release()

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(sampling_freq),
            '-i', os.path.join(temp_frame_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            processed_video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        shutil.rmtree(temp_frame_dir)

        final_tokens_per_frame = calculate_tokens(new_width, new_height)
        metadata = {
            **entry,
            'video': os.path.join(relative_path, video_filename),
            'video_metadata': {
                'original': {
                    'duration': total_duration,
                    'frame_count': total_frames,
                    'fps': fps,
                    'resolution': {'width': orig_width, 'height': orig_height},
                    'tokens_per_frame': tokens_per_frame
                },
                'processed': {
                    'frame_count': frames_extracted,
                    'sampling_frequency': sampling_freq,
                    'sampling_mode': 'uniform' if total_duration > max_frames else '1fps',
                    'resolution': {'width': new_width, 'height': new_height},
                    'tokens_per_frame': final_tokens_per_frame,
                    'total_tokens': final_tokens_per_frame * frames_extracted
                }
            }
        }
        return metadata

    except Exception as e:
        print(f"Error processing video {entry.get('video', 'unknown')}: {e}")
        if 'temp_frame_dir' in locals() and os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir)
        return None

def load_and_process_data(ds_name, base_path, output_path, max_duration, max_tokens, max_frames, num_processes):
    ensure_dir(output_path)

    if ds_name.lower() == "openbiomedvid":
        dataset = load_dataset("connectthapa84/OpenBiomedVid", split="train")
    elif ds_name.lower() == "surgeryvideoqa":
        dataset = load_dataset("connectthapa84/SurgeryVideoQA", split="test")
    else:
        raise ValueError("Dataset must be 'OpenBiomedVid' or 'SurgeryVideoQA'.")

    unique_videos = {entry['video']: entry for entry in dataset}
    print(f"\nProcessing {len(unique_videos)} unique videos using {num_processes} processes...")

    func = partial(process_video,
                   base_path=base_path,
                   output_path=output_path,
                   max_duration=max_duration,
                   max_tokens=max_tokens,
                   max_frames=max_frames)

    with mp.Pool(num_processes) as pool:
        processed_videos = list(tqdm(
            pool.imap_unordered(func, unique_videos.values()),
            total=len(unique_videos),
            desc="Processing videos"
        ))
        data = [video for video in processed_videos if video is not None]

    print(f"\nSuccessfully processed {len(data)} videos")
    return data

def save_metadata(data, output_jsonl):
    print(f"\nSaving metadata to {output_jsonl}")
    with open(output_jsonl, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Path to input videos")
    parser.add_argument("--dataset", type=str, required=True, choices=["OpenBiomedVid", "SurgeryVideoQA"], help="Dataset to process.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed videos")
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes to use")
    parser.add_argument("--max_frames", type=int, default=420, help="Maximum number of frames")
    parser.add_argument("--max_duration", type=int, default=420, help="Max duration in seconds")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max allowed tokens per video")
    parser.add_argument("--output_jsonl", type=str, default="openbiomedvid_final.jsonl", help="Name for output JSONL")

    args = parser.parse_args()

    data = load_and_process_data(
        args.dataset,
        args.base_path,
        args.output_path,
        args.max_duration,
        args.max_tokens,
        args.max_frames,
        args.num_processes
    )

    os.makedirs("data", exist_ok=True)
    save_metadata(data, os.path.join("data", args.output_jsonl))
    print("Done!")

if __name__ == "__main__":
    main()
