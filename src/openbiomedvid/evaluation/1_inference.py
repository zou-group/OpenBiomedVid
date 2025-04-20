# %%
import sys
import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

logger = logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_pretrained_qwen(model_path, device):
    """Load the Qwen model and processor."""
    print(f"Loading Qwen model from {model_path} on device {device}")

    if "Qwen2-VL" in model_path or "qwen2_vl" in model_path.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map={"": device},  # Map all modules to the specific device
            use_flash_attention_2=True,
            attn_implementation="flash_attention_2"
        )
        if "Qwen2-VL-7B-Instruct" in model_path or "qwen2_vl_7b_instruct" in model_path.lower():
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        elif "Qwen2-VL-2B-Instruct" in model_path or "qwen2_vl_2b_instruct" in model_path.lower():
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        elif "Qwen2-VL-72B-Instruct" in model_path or "qwen2_vl_72b_instruct" in model_path.lower():
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
        else:
            raise ValueError(f"Processor for model {model_path} not found.")
    else:
        raise ValueError(f"Model {model_path} not supported.")
    return model, processor


def load_data(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data  # Return list directly, don't convert to Dataset

def generate_task(args):
    """Run inference on the dataset using Qwen2-VL."""
    # Setup distributed training
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    
    if local_rank == 0:
        print(f"Running on {world_size} GPUs")
        print(f"Task: {args.task_name}")
        print(f"Model: {args.model_path}")

    if args.debug and local_rank == 0:
        logger.info(f"Running in Debug Mode.")

    model, processor = load_pretrained_qwen(args.model_path, device)
    max_new_tokens = args.max_new_tokens

    # Load dataset
    if args.task_name.lower() == "mimicechoqa":
        dataset = list(load_dataset("connectthapa84/MIMICEchoQA", split="test"))
    elif args.task_name.lower() == "surgeryvideoqa":
        dataset = list(load_dataset("connectthapa84/SurgeryVideoQA", split="test"))
    else:
        raise ValueError("Dataset must be 'OpenBiomedVid' or 'SurgeryVideoQA'.")

    items_per_gpu = len(dataset) // world_size
    start_idx = local_rank * items_per_gpu
    end_idx = start_idx + items_per_gpu if local_rank < world_size - 1 else len(dataset)
    gpu_dataset = dataset[start_idx:end_idx]

    if local_rank == 0:
        print(f"Total samples: {len(dataset)}")
        print(f"Samples per GPU: ~{items_per_gpu}")

    # Setup output files
    model_name = args.model_path.split("/")[-1]
    path_to_save = os.path.join(args.output_path, f"logs/{args.task_name}")
    os.makedirs(path_to_save, exist_ok=True)
    
    # Create a temporary file for this GPU's results
    temp_file = os.path.join(path_to_save, f"temp_results_rank_{local_rank}.jsonl")
    results = []  # Store results in memory

    # Process this GPU's portion of the data
    for i, line in enumerate(tqdm(gpu_dataset, disable=local_rank != 0)):
        answer = line['answer']
        question = line['question']

        if args.task_name == "mimic_echo":
            # question = f"Question: {question}\nOptions: A: {line['option_A']}\nB: {line['option_B']}\nC: {line['option_C']}\nD: {line['option_D']}\nChoose A, B, C, or D."
            if answer.lower() in ["yes", "no"]:
                question = f"Question: {question}\nChoose Yes or No."
                # question = f"Question: {question}"
            else:
                question = f"Question: {question}\nOptions: A: {line['option_A']}\nB: {line['option_B']}\nC: {line['option_C']}\nD: {line['option_D']}\nChoose A, B, C, or D."

        video_path = os.path.join(args.task_video_path, line['video'])
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": question}]}]
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=video_inputs, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                use_cache=True)
        
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
        response = []
        end_tokens = ["<|im_end|>", "</s>", "<|eot_id|>"]
        for gen_text in outputs:
            gen_text_new = gen_text.split("assistant")[-1].strip(" ").strip("\n")
            for end_token in end_tokens:
                gen_text_new = gen_text_new.split(end_token)[0]
            response.append(gen_text_new)

        pred = response[0].strip()

        result_dict = {
            "prompt": text, 
            "pred": pred, 
            "model_id": args.model_path,
            "original_index": start_idx + i  # Keep track of original position
        }
        result_dict.update(line)

        if "image" in result_dict:
            del result_dict["image"]
            
        results.append(result_dict)

    # Save temporary results
    with open(temp_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Wait for all processes to finish
    dist.barrier()

    # Combine results on rank 0
    if local_rank == 0:
        all_results = []
        # Gather results from all GPUs
        for rank in range(world_size):
            rank_file = os.path.join(path_to_save, f"temp_results_rank_{rank}.jsonl")
            with open(rank_file, 'r') as f:
                rank_results = [json.loads(line) for line in f]
                all_results.extend(rank_results)

        # Sort by original index to maintain order
        all_results.sort(key=lambda x: x.pop('original_index'))

        # Save final combined results
        final_output = os.path.join(path_to_save, "results.jsonl")
        with open(final_output, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        # Clean up temporary files
        for rank in range(world_size):
            temp_file = os.path.join(path_to_save, f"temp_results_rank_{rank}.jsonl")
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print(f"Inference complete. Results saved to {final_output}")
        print(f"Total processed samples: {len(all_results)}")

    # Final barrier to ensure cleanup is complete
    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--task_name", type=str, default="SurgeryVideoQA", choices=["SurgeryVideoQA", "MIMICEchoQA"])
    parser.add_argument("--task_video_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=float, default=32)
    parser.add_argument("--output_path", type=str, default="/data/rahulthapa/logs")
    parser.add_argument("--debug", action="store_true", help="Activate debug mode")
    parser.add_argument("--device", type=int, help="Device Type", default=0)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    generate_task(args)