import os
import matplotlib.pyplot as plt
import json
import shutil
from tqdm import tqdm

def get_video_list(video_dir):
    """
    Returns a list of all MP4 video paths in the directory and its subdirectories.
    
    Args:
        video_dir (str): Path to the directory containing videos
        
    Returns:
        list: List of full paths to MP4 files
    """
    video_list = []
    
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_list.append(video_path)
    
    print(f"Found {len(video_list)} videos in {video_dir}")
    return video_list

def get_video_info(video_path):
    """
    Returns the name, start time, and end time of a video.
    """
    parts = os.path.basename(video_path).replace('.mp4', '').split('_')
    video_id = '_'.join(parts[:-2])
    start_time = int(parts[-2])
    end_time = int(parts[-1])
    return video_id, start_time, end_time

def get_captions(caption_path, start_time, end_time):
    """
    Returns the captions for a video in a given time range.
    """
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)

        chunks = caption_data['chunks']
        
        filtered_chunks = []
        for chunk in chunks:
            # Skip if both timestamps are None
            if not chunk['timestamp'][0] and not chunk['timestamp'][1]:
                continue
                
            # If first timestamp is None, set it to second - 0.01
            if not chunk['timestamp'][0]:
                chunk['timestamp'][0] = chunk['timestamp'][1] - 0.01
                
            # If second timestamp is None, set it to first + 0.01 
            if not chunk['timestamp'][1]:
                chunk['timestamp'][1] = chunk['timestamp'][0] + 0.01
                
            # Keep chunk if timestamps are within start/end time range
            if float(start_time) <= chunk['timestamp'][0] <= float(end_time) and \
                float(start_time) <= chunk['timestamp'][1] <= float(end_time):
                filtered_chunks.append(chunk)

    return filtered_chunks

def plot_video_lengths(video_dir):
    """
    Plots the distribution of video lengths and prints statistics.
    """
    # Get list of all videos
    videos = get_video_list(video_dir)
    
    # Calculate lengths
    lengths = []
    for video in videos:
        _, start_time, end_time = get_video_info(video)
        length = end_time - start_time
        lengths.append(length)
    
    # Calculate statistics
    total_length = sum(lengths)
    hours = total_length / 3600
    print(f"\nVideo Statistics:")
    print(f"Total videos: {len(lengths)}")
    print(f"Total length: {hours:.2f} hours")
    print(f"Mean length: {sum(lengths)/len(lengths):.2f} seconds")
    
    # Create histogram
    plt.figure(figsize=(10,6))
    plt.hist(lengths, bins=50)
    plt.xlabel('Video Length (seconds)')
    plt.ylabel('Count')
    plt.title('Distribution of Video Lengths')
    plt.savefig('video_length_distribution for {video_dir}.png')
    plt.close()

def copy_videos(source_dir, dest_dir):
    """
    Copies all MP4 videos from source directory to destination directory.
    Maintains the same filename but flattens the directory structure.
    
    Args:
        source_dir (str): Source directory containing videos
        dest_dir (str): Destination directory for copied videos
    """
    # Get list of all videos
    print(f"Copying videos from {source_dir} to {dest_dir}")
    video_paths = get_video_list(source_dir)
    os.makedirs(dest_dir, exist_ok=True)

    for video_path in tqdm(video_paths):
        video_name = os.path.basename(video_path)
        category = os.path.basename(os.path.dirname(video_path))
        dest_path = os.path.join(dest_dir, category, video_name)

        if os.path.exists(dest_path):
            continue

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(video_path, dest_path)