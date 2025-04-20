import argparse
import json
import os
import random
import uuid
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, DatasetDict
from tqdm import tqdm

VIDEO_PROMPTS = [
    "Please describe the biomedical content shown in this video.",
    "What medical or clinical content can you observe in this video?",
    "Could you explain the medical aspects shown in this footage?",
    "Please provide a description of the medical content demonstrated in this video.",
    "What biomedical information is being presented in this video?",
    "Can you describe the medical content shown in this footage?",
    "Please explain what you observe in this medical video.",
    "What medical or clinical elements are demonstrated in this video?",
    "Could you describe the biomedical content presented here?",
    "Please detail the medical information shown in this video.",
    "What do you observe in this medical footage?",
    "Can you explain the biomedical content demonstrated here?",
    "Please describe what's being shown in this medical video.",
    "What medical content is being presented in this footage?",
    "Could you detail the biomedical aspects shown in this video?",
    "Please explain the medical elements demonstrated here.",
    "What clinical or medical content do you observe in this video?",
    "Can you describe the biomedical information shown in this footage?",
    "Please provide an explanation of the medical content in this video.",
    "What medical or clinical aspects are being demonstrated here?"
]

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
    return data

def create_caption_entry(item: Dict, base_path: str) -> Dict:
    source = Path(item['video']).parts[0]
    messages = [
        {"role": "user", "content": f"<|video|>\n{random.choice(VIDEO_PROMPTS)}"},
        {"role": "assistant", "content": item["caption"]}
    ]
    return {
        "source": source,
        "message_id": str(uuid.uuid4()),
        "messages": messages,
        "videos": [os.path.join(base_path, item['video'])],
        "frame_count": item["video_metadata"]["processed"]["frame_count"],
        "frame_resolution": item["video_metadata"]["processed"]["resolution"],
        "sampling_frequency": item["video_metadata"]["processed"]["sampling_frequency"],
        "special_tokens": ["<|video|>"]
    }

def create_qa_entry(item: Dict, base_path: str) -> List[Dict]:
    source = Path(item["video"]).parts[0]
    messages = [{"role": "user", "content": "<|video|>"}]
    for qa_pair in item["qa_pairs"]:
        messages.append({"role": "user", "content": qa_pair["question"]})
        messages.append({"role": "assistant", "content": qa_pair["answer"]})
    return [{
        "source": source,
        "message_id": str(uuid.uuid4()),
        "messages": messages,
        "videos": [os.path.join(base_path, item["video"])],
        "frame_count": item["video_metadata"]["processed"]["frame_count"],
        "frame_resolution": item["video_metadata"]["processed"]["resolution"],
        "sampling_frequency": item["video_metadata"]["processed"]["sampling_frequency"],
        "special_tokens": ["<|video|>"]
    }]

def generate_dataset(jsonl_path: str, base_path: str) -> List[Dict]:
    raw_data = load_jsonl(jsonl_path)
    dataset = []
    for item in tqdm(raw_data, desc="Generating dataset"):
        if "caption" not in item:
            continue
        try:
            dataset.append(create_caption_entry(item, base_path))
            if "qa_pairs" in item and item["qa_pairs"]:
                dataset.extend(create_qa_entry(item, base_path))
        except KeyError as e:
            print(f"Skipping entry due to missing key: {e}")
    return dataset

def process_video_meta(example):
    video_metas = [{
        "num_frames": example["frame_count"],
        "resolution": example["frame_resolution"],
        "fps": example["sampling_frequency"]
    }]
    example["video_metas"] = json.dumps(video_metas)
    return example

def main():
    parser = argparse.ArgumentParser(description="Prepare biomedical video dataset for instruction tuning.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the video segments.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")
    parser.add_argument("--num_proc", type=int, default=64, help="Number of processes for multiprocessing map.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name (e.g., train, valid, test).")

    args = parser.parse_args()

    # Step 1: Generate dataset
    dataset_entries = generate_dataset(args.jsonl_path, args.base_path)

    # Step 2: Convert to HuggingFace DatasetDict
    hf_dataset = Dataset.from_list(dataset_entries)
    dataset_dict = DatasetDict({args.split: hf_dataset})

    # Step 3: Process video metadata and clean
    dataset_dict = dataset_dict.map(process_video_meta, num_proc=args.num_proc)
    dataset_dict = dataset_dict.remove_columns(["frame_count", "frame_resolution", "sampling_frequency"])
    dataset_dict = dataset_dict.select_columns(['source', 'message_id', 'messages', 'videos', 'video_metas', 'special_tokens'])

    # Step 4: Save
    print(f"Saving to: {args.output_dir}")
    dataset_dict.save_to_disk(args.output_dir)
    print("✅ Dataset preparation complete!")

if __name__ == "__main__":
    main()
