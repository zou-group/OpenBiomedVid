import os
import argparse
import numpy as np
from datasets import load_dataset
from moviepy.editor import VideoFileClip
from multiprocessing import Pool
from tqdm import tqdm
import tempfile

def process_example(example_and_paths):
    import tempfile

    example, input_video_dir, output_video_dir = example_and_paths

    video_id = example["video_id"]
    video_filename = f"{video_id}.mp4"
    sliced_filename = example["video"]
    timestamp = example["timestamp"]

    input_path = os.path.join(input_video_dir, video_filename)
    output_path = os.path.join(output_video_dir, sliced_filename)
    temp_audio_path = os.path.join(tempfile.gettempdir(), f"{sliced_filename}_temp_audio.m4a")

    video = None
    segment = None

    if not os.path.exists(input_path):
        return f"[Warning] Missing video: {video_filename}"

    try:
        video = VideoFileClip(input_path)
        segment = video.subclip(timestamp[0], timestamp[1])

        segment.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            threads=4,
            temp_audiofile=temp_audio_path,
            remove_temp=True
        )
        return f"[Success] {sliced_filename}"

    except Exception as e:
        return f"[Error] {sliced_filename} - {e}"

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if segment is not None:
            segment.close()
        if video is not None:
            video.close()


def main(dataset_name, input_video_dir, output_video_dir, num_processes):
    # Load the appropriate split from HF dataset
    if dataset_name.lower() == "openbiomedvid":
        dataset = load_dataset("connectthapa84/OpenBiomedVid", split="train")
    elif dataset_name.lower() == "surgeryvideoqa":
        dataset = load_dataset("connectthapa84/SurgeryVideoQA", split="test")
    else:
        raise ValueError("Dataset must be 'OpenBiomedVid' or 'SurgeryVideoQA'.")

    os.makedirs(output_video_dir, exist_ok=True)


    # Split into chunks
    dataset_chunks = np.array_split(dataset, num_processes)
    args_list = [
        (example, input_video_dir, output_video_dir)
        for chunk in dataset_chunks for example in chunk
    ]

    print(f"Processing {len(args_list)} videos using {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_example, args_list), total=len(args_list)):
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel video slicing using moviepy + HuggingFace datasets.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["OpenBiomedVid", "SurgeryVideoQA"],
                        help="Dataset to process.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing full videos.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save sliced video segments.")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of parallel processes.")

    args = parser.parse_args()
    main(args.dataset, args.input_dir, args.output_dir, args.num_processes)
