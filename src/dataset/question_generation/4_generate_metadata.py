import argparse
import os
import json
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
import concurrent.futures

client = OpenAI()

SYSTEM_PROMPT = """
You are an expert in biomedical video analysis and medical diagnosis. You are tasked with labeling the modality of the video and anatomical region of the video given the video caption. You must choose from the following options:

Modality:
    - Echocardiography
    - Ultrasound
    - CT Imaging
    - MRI
    - Angiography
    - X-ray
    - Fluoroscopy
    - Endoscopy and Laparoscopic Surgery
    - Surgical and Procedural
    - Anatomy and Biological Processes
    - Microscopy
    - Other

Anatomical Region:
    - Cardiac
    - Vascular System
    - Musculoskeletal System
    - Cranial and Nervous System
    - Thoracic and Respiratory System
    - Gastrointestinal System
    - Genitourinary System
    - Head and Neck
    - Endocrine System
    - Skin and Integumentary System
    - Other 
"""

class Category(BaseModel):
    modality: str
    anatomical_region: str

# Function to process a single caption and generate Q/A pairs
def process_single_qa_pairs(sample):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample['cleaned_caption']}
            ],
            response_format=Category,
        )

        qa_data = completion.choices[0].message.parsed
        # Create output data structure - retain all original keys and add qa_pairs
        output_data = {
            **sample,  # Include all original keys
            "metadata": {
                "modality": qa_data.modality,
                "anatomical_region": qa_data.anatomical_region
            }
        }
        return output_data

    except Exception as e:
        print(f"Error processing video {sample['video_id']}: {str(e)}")
        return None  # Return None if an error occurs

# Parallel processing function
def process_qa_pairs_parallel(input_path, output_path, max_workers=8):
    # Read input JSONL file
    with open(input_path, 'r') as f:
        samples = [json.loads(line) for line in f]

    # Ensure output path exists and clear existing file if necessary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # if os.path.exists(output_path):
    #     os.remove(output_path)

    # Process captions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with open(output_path, 'a') as out_f:
            futures = {
                executor.submit(process_single_qa_pairs, sample): sample
                for sample in samples
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples), desc="Processing captions"):
                result = future.result()
                json.dump(result, out_f)
                out_f.write('\n')
                out_f.flush()  # Ensure it's written to disk

# Main function
def main(args):
    input_path = args.input_path
    output_path = args.output_dir
    max_workers = args.max_workers

    print("Input path:", input_path)
    print("Output path:", output_path)
    print("Max workers:", max_workers)

    # Run parallel processing
    process_qa_pairs_parallel(input_path, output_path, max_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input JSONL file containing cleaned captions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to file to save Q/A pairs (not just directory)')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Maximum number of parallel threads')
    args = parser.parse_args()

    main(args)