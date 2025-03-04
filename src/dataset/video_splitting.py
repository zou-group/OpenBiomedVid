import os
import argparse
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import get_video_list, get_captions, get_video_info

client = OpenAI()

class Split(BaseModel):
    original_segment: List[float] = Field(..., description="Start and end timestamps of the original segment")
    timestamps: List[List[float]] = Field(..., description="List of timestamps to split the original segment into")
    reasons: List[str] = Field(..., description="List of reasons for the splits")

SYSTEM_PROMPT = """You are a video content analyzer. Your task is to analyze video transcript segments provided as a list of JSON objects, where each object contains 'words' (string of transcript words) and 'timestamp' [start_time, end_time]. You should concatenate the words together to understand the full context while using the timestamps to determine segment boundaries. Your tasks are:

1. Identify natural breakpoints where the content can be split into smaller, self-contained segments by looking for:
   - Topic transitions
   - Changes in subject matter
   - Natural pauses or breaks in the discussion
   - Completion of a concept or idea

2. For each suggested split, provide:
   - The reason for the split
   - The new timestamp ranges for the resulting segments

Focus on creating segments that maintain coherent context and are logically self-contained. Each new segment MUST be between 120-300 seconds (2-5 minutes) in length, meaning the difference between the end and start timestamps must be between 120 and 300 seconds. All timestamps must be non-negative numbers and must be between the start and end time of the video and the given words.

Example input format: [{'text': 'first part of sentence', 'timestamp': [120.5, 180.3]}, {'text': 'second part of sentence', 'timestamp': [180.3, 250.2]}, ...]

Example output format:
'original_segment': [120.5, 600.8]
'timestamps': [
    [120.5, 250.2],
    [250.2, 420.5],
    [420.5, 600.8]
],
'reasons': [
    'Introduction of concept A',
    'Detailed explanation of A', 
    'Transition to new topic'
]"""

def process_video(video_info, audio_dir, output_dir):
    video_path, video_id, start_time, end_time = video_info

    category = os.path.basename(os.path.dirname(video_path))
    captions = os.path.join(audio_dir, f"{category}/{video_id}.json")
    
    try:
        chunks = get_captions(captions, start_time, end_time)

        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"The transcription is here: {chunks}. The START time is {start_time} and the END time is {end_time}."}
            ],
            response_format=Split
        )

        split = completion.choices[0].message.parsed

        for timestamp in split.timestamps:
            start_segment = round(timestamp[0] + 0.5)
            end_segment = round(timestamp[1] - 0.5)

            if start_segment >= end_segment or \
                start_segment < start_time or \
                end_segment > end_time:
                continue

            new_video_name = f"{video_id}_{start_segment}_{end_segment}.mp4"
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)
            new_video_path = os.path.join(output_dir, category, new_video_name)

            video = VideoFileClip(video_path)
            segment = video.subclip(start_segment - start_time, end_segment - start_time)

            os.makedirs(os.path.join(output_dir, 'temp'), exist_ok=True)
            temp_audio_path = os.path.join(output_dir, 'temp', f"temp-audio-{new_video_name.replace('.mp4', '.mp3')}")

            try:
                segment.write_videofile(
                    new_video_path,
                    codec="libx264",
                    logger=None,
                    threads=4,
                    temp_audiofile=temp_audio_path
                )
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                segment.close()
                video.close()
        
        return f"Successfully processed {video_id}"
    except Exception as e:
        return f"Error processing {video_id}: {str(e)}"

def main(args):
    video_dir = args.video_dir
    audio_dir = args.audio_dir
    output_dir = args.output_dir

    # Get all long videos > 5 minutes
    long_videos = []
    videos = get_video_list(video_dir)

    for video_path in videos:       
        video_id, start_time, end_time = get_video_info(video_path)

        if end_time - start_time > 300:
            long_videos.append((video_path, video_id, start_time, end_time))

    max_workers = min(os.cpu_count(), 128)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
       
        future_to_video = {
            executor.submit(process_video, video_info, audio_dir, output_dir): video_info 
            for video_info in long_videos
        }
        
       
        for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="Processing videos"):
            video_info = future_to_video[future]
            try:
                result = future.result()
                
            except Exception as e:
                print(f"Error in executor for {video_info[1]}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Path to directory containing relevant video segments')
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to directory containing complete audio files')
    parser.add_argument('--output_dir', type=str, required=False, help='Path to directory to save split videos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)


        



