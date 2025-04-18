import argparse
import os
from tqdm import tqdm
import json
from pydantic import BaseModel
from openai import OpenAI
import concurrent.futures

client = OpenAI()

SYSTEM_PROMPT = """
Your task is to refine text by improving grammar, clarity, and professionalism while preserving all original details. The text comes from a **biomedical video caption**, so it should be rewritten in a **descriptive format** that objectively narrates the video content.  

### Guidelines:  
- Correct grammar and sentence structure for better readability.  
- Ensure a professional and scientific tone without changing the meaning.  
- Rewrite first-person references into a formal, objective description of the video.  
- Format the text as if it is describing the biomedical video's content rather than direct speech or narration.  
- Do not add, remove, or modify information beyond the original text.  
- Avoid hallucinations—stick strictly to the provided content.  

Output should be polished, objective, and well-structured while maintaining accuracy.
"""

class Caption(BaseModel):
    cleaned_caption: str

def load_jsonl(input_path):
    """
    Loads a JSONL file into a list of dictionaries.
    """
    with open(input_path, 'r') as f:
        return [json.loads(line) for line in f]

def process_single_caption(video_data):
    """
    Processes a single caption using GPT to refine and clean it.
    """
    try:
        caption = video_data['caption']
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": caption}
            ],
            response_format=Caption,
        )

        caption_data = completion.choices[0].message.parsed
        video_data['cleaned_caption'] = caption_data.cleaned_caption
        return video_data
    except Exception as e:
        print(f"Error processing video {video_data.get('video_id', 'unknown')}: {str(e)}")
        video_data['cleaned_caption'] = caption  # Save original caption if error occurs
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
    # if os.path.exists(output_path):
    #     os.remove(output_path)

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
