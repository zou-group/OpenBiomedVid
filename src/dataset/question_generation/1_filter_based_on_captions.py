import argparse
import os
from tqdm import tqdm
import json
from pydantic import BaseModel
from openai import OpenAI
import concurrent.futures

client = OpenAI()

SYSTEM_PROMPT = """You are an AI assistant that determines if a caption describes relevant biomedical video content. 

The caption must meet BOTH criteria to be considered biomedical:

1. Describes observable visual content:
- Medical imaging (ultrasounds, X-rays, MRI, CT scans, etc.)
- Clinical procedures or examinations
- Surgical operations
- Microscopic views
- Anatomical demonstrations
- Medical device operations
- Patient assessments
- Laboratory procedures with visual components

2. Contains specific biomedical elements:
- Not just medical settings or personnel
- Must describe actual medical/biological content
- Should reference visual elements that would be seen in the video
}"""

class RelevantCaption(BaseModel):
    is_biomedical: bool

def load_jsonl(input_path):
    """
    Loads a JSONL file into a list of dictionaries.
    """
    with open(input_path, 'r') as f:
        return [json.loads(line) for line in f]

def process_single_caption(video_data):
    """
    Processes a single caption using GPT to determine if it is biomedical.
    """
    try:
        caption = video_data['caption']
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": caption}
            ],
            response_format=RelevantCaption,
        )

        caption_data = completion.choices[0].message.parsed
        video_data['is_biomedical'] = caption_data.is_biomedical
        return video_data
    except Exception as e:
        print(f"Error processing video {video_data.get('video_id', 'unknown')}: {str(e)}")
        video_data['is_biomedical'] = False  # Save original caption if error occurs
        return video_data

def process_captions_parallel(input_path, output_path, max_workers=64):
    """
    Processes captions in parallel using multithreading.
    """
    data = load_jsonl(input_path)

    clean_data = []

    for item in data:
        caption = item['caption']
        if len(caption.split(' ')) < 10:
            continue
        clean_data.append(item)
    
    data = clean_data

    # Clear output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Process captions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with open(output_path, 'a') as out_f:
            futures = {
                executor.submit(process_single_caption, video_data): video_data
                for video_data in data
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Processing captions"):
                result = future.result()
                json.dump(result, out_f)
                out_f.write('\n')
                out_f.flush()  # Ensure it's written to disk
    
    print("Filtering out non-biomedical captions")
    # Filter out non-biomedical captions
    with open(output_path, 'r') as f:
        data = [json.loads(line) for line in f]

    data = [item for item in data if item['is_biomedical']]

    with open(output_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

    print(f"Filtered {len(data)} captions out of {len(data)}")

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    max_workers = args.max_workers

    print("Input path:", input_path)
    print("Output path:", output_path)
    print("Max workers:", max_workers)

    process_captions_parallel(input_path, output_path, max_workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to input jsonl file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output jsonl file')
    parser.add_argument('--max_workers', type=int, default=64, help='Maximum number of parallel threads')
    args = parser.parse_args()

    main(args)
