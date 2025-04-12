import numpy as np
import os
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from together import Together
from moviepy.editor import VideoFileClip, AudioFileClip
from openai import OpenAI
from pydantic import BaseModel
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import inspect
import textwrap
import argparse

class MedicalSegmentAnalyzer:
    def __init__(self, 
                 npy_dir: str,
                 audio_dir: str,
                 video_dir: str,
                 output_dir: str,
                 json_file: str,
                 confidence_threshold: float):
        self.npy_dir = npy_dir
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.json_file = json_file
        self.confidence_threshold = confidence_threshold
        self.max_gap = 10
        self.output_file = os.path.join(self.output_dir, self.json_file)
        
        # Create set of processed video IDs
        self.processed_videos = set()
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                for line in f:
                    segment = json.loads(line)
                    self.processed_videos.add(segment['id'])
        
        print(f"Loaded {len(self.processed_videos)} existing processed videos")

    def find_high_confidence_segments(self, 
                                    predictions: np.ndarray,
                                    max_gap: Optional[int] = None, 
                                    buffer: Optional[int] = 0) -> List[Tuple[float, float]]:
        """
        Find segments using a sliding window approach with gap tolerance.
        
        Args:
            predictions: numpy array of (timestamp, probability) pairs
            max_gap: maximum number of consecutive negative frames to tolerate
            
        Returns:
            List of (start_time, end_time) tuples for segments
        """
        segments = []
        current_segment_start = None
        gap_count = 0

        if max_gap is None:
            max_gap = self.max_gap

        min_time = 0
        max_time = predictions[-1][0]
        
        for i, (timestamp, prob) in enumerate(predictions):
            # Case 1: High confidence frame
            if prob > self.confidence_threshold:
                if current_segment_start is None:
                    current_segment_start = timestamp
                gap_count = 0
                
            # Case 2: Low confidence frame
            else:
                if current_segment_start is not None:
                    gap_count += 1
                    
                    # If gap is too large, close current segment
                    if gap_count > max_gap:
                        # Use the timestamp from max_gap frames ago as end
                        end_timestamp = predictions[i - gap_count][0]
                        current_segment_start = max(min_time, current_segment_start - buffer)
                        end_timestamp = min(max_time, end_timestamp + buffer)
                        segments.append((current_segment_start, end_timestamp))
                        current_segment_start = None
                        gap_count = 0
        
        # Handle the last segment if it exists
        if current_segment_start is not None:
            if gap_count > 0:
                end_timestamp = predictions[len(predictions) - gap_count - 1][0]
            else:
                end_timestamp = predictions[-1][0]
            current_segment_start = max(min_time, current_segment_start - buffer)
            end_timestamp = min(max_time, end_timestamp + buffer)
            segments.append((current_segment_start, end_timestamp))
        
        return segments

    def get_captions_for_intervals(self, 
                                 words_json: Dict, 
                                 intervals: List[Tuple[float, float]]) -> List[Dict]:
        """Extract captions for specific time intervals."""
        captions = []
        
        for start_time, end_time in intervals:
            audio_start = start_time
            audio_end = end_time

            segment_words = []
            
            for chunk in words_json['chunks']:
                word_start, word_end = chunk['timestamp']
                
                # Handle null timestamps
                if word_start is None and word_end is None:
                    continue
                elif word_start is None:
                    word_start = word_end - 0.01
                elif word_end is None:
                    word_end = word_start + 0.01
                
                if (word_start >= audio_start and word_start < audio_end) or \
                   (word_end > audio_start and word_end <= audio_end):
                    segment_words.append(chunk['text'])
            
            if segment_words:
                caption = ' '.join(segment_words).strip()
                captions.append({
                    'text': caption,
                    'timestamp': [start_time, end_time]
                })
        
        return captions

    def analyze_video(self, npy_path: str) -> Optional[Dict]:
        """Analyze a single video's high confidence segments."""
        # Get video ID and category from full path
        video_id = os.path.basename(npy_path).split('.')[0]
        category = os.path.basename(os.path.dirname(npy_path))
        
        # Check if already processed using the set
        if video_id in self.processed_videos:
            print(f"Skipping {category}/{video_id} - already processed")
            return None
        
        # Create output directory for this category
        os.makedirs(os.path.join(self.output_dir, category), exist_ok=True)
        
        # Get input paths - caption json should be directly in audio category folder
        input_words_path = os.path.join(self.audio_dir, category, f"{video_id}.json")
        
        if not (os.path.exists(input_words_path) or os.path.exists(npy_path)):
            print(f"Missing file for {category}/{video_id}")
            return
        
        try:
            predictions = np.load(npy_path)
            if len(predictions) == 0:
                print(f"Empty predictions file for {category}/{video_id}")
                return
        except (EOFError, ValueError) as e:
            print(f"Error loading predictions for {category}/{video_id}: {str(e)}")
            return
        
        with open(input_words_path, 'r') as f:
            words_json = json.load(f)
        
        high_conf_intervals = self.find_high_confidence_segments(predictions, buffer=0)
        high_conf_captions = self.get_captions_for_intervals(words_json, high_conf_intervals)

        if len(high_conf_captions) == 0:
            print(f"No high confidence segments found for {category}/{video_id}")
            return

        result = []
        
        for caption in high_conf_captions:
            start_time, end_time = caption['timestamp']

            start_time = float(start_time)
            end_time = float(end_time)
            
            video_name = f"{video_id}_{round(start_time)}_{round(end_time)}.mp4"
            video_output = os.path.join(self.output_dir, category, video_name)

            temp_dir = os.path.join(self.output_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_audio_path = os.path.join(temp_dir, f"temp-audio-{video_name.replace('.mp4', '.mp3')}")
            
            video_path = os.path.join(self.video_dir, category, f"{video_id}.mp4")
            video = VideoFileClip(video_path)
            segment = video.subclip(start_time, end_time)

            try:

                segment.write_videofile(
                    video_output,
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
                    
            result.append({
                'id': video_id,
                'caption': caption['text'],
                'timestamp': [float(start_time), float(end_time)],
                'video': os.path.join(category, video_name),
                'data_source': os.path.join(category, video_id)
            })
        
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Segment Analyzer')
    parser.add_argument('--npy_dir', type=str, 
                       default="/data/andrew/biomed_yt_videos/frame_outputs/train_1/",
                       help='Directory containing numpy files')
    parser.add_argument('--audio_dir', type=str,
                       default="/data/andrew/biomed_yt_videos/audio_outputs/",
                       help='Directory containing audio files')
    parser.add_argument('--video_dir', type=str,
                       default="/data/rahulthapa/biomed_yt_videos/",
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str,
                       default="/data/andrew/biomed_yt_videos/video_outputs/train_1/",
                       help='Output directory for processed videos')
    parser.add_argument('--json_file', type=str,
                       default="medical_segments.jsonl",
                       help='Output JSON file name')
    parser.add_argument('--confidence_threshold', type=float,
                       default=0.64,
                       help='Confidence threshold for segment selection')
    parser.add_argument('--split', type=str,
                       choices=['first', 'second', 'third', 'all'],
                       default='all',
                       help='Which portion of data to process (first, second, third, or all)')
    parser.add_argument('--debug', type=int,
                       default=0,
                       help='Debug mode')
    
    args = parser.parse_args()
    
    analyzer = MedicalSegmentAnalyzer(
        npy_dir=args.npy_dir,
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        json_file=args.json_file,
        confidence_threshold=args.confidence_threshold
    )

    npy_files = []
    
    # Get all NPY files including subfolders
    npy_files = []
    for root, _, files in os.walk(analyzer.npy_dir):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))

    print("output_dir", analyzer.output_dir)
    random.shuffle(npy_files)

    max_workers = 128

    # Calculate splits
    total_files = len(npy_files)
    split_size = total_files // 3
    
    splits = {
        'first': npy_files[:split_size],
        'second': npy_files[split_size:split_size*2],
        'third': npy_files[split_size*2:],
        'all': npy_files
    }
    
    # Use the specified split
    npy_files = splits[args.split]
    print(f"Processing {len(npy_files)} files from {args.split} portion")
    print(f"Total files: {total_files}")

    output_file = os.path.join(analyzer.output_dir, analyzer.json_file)
    print(f"Results saving to {output_file}")

    if not os.path.exists(output_file):
        open(output_file, 'w').close()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(analyzer.analyze_video, npy_file): npy_file 
            for npy_file in npy_files
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(npy_files), desc="Analyzing videos"):
            npy_file = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    with open(output_file, 'a') as f:
                        for item in result:
                            json.dump(item, f)
                            f.write('\n')
                    
            except Exception as e:
                print(f"Error processing {npy_file}: {str(e)}")
    
    print(f"\nAnalyzed {len(npy_files)} videos")
    print(f"Number of lines in output file: {sum(1 for _ in open(output_file))}")