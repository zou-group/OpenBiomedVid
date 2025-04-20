"""
Supervised fine-tuning script for decoder language models.
"""

from typing import Any, Dict, List, Optional
import copy
import gc
import os
import logging
import random
import sys
import tqdm
from torch.nn.utils.rnn import pad_sequence
from dataclasses import asdict, dataclass, field
from io import BytesIO
import datasets
from accelerate.state import PartialState
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import set_seed
from transformers.trainer_utils import seed_worker
from datasets import load_dataset, load_from_disk, concatenate_datasets

import os
import torch
import json
import numpy as np
from PIL import Image


# Tore Train Imports
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from tore_train.trainers import ToreTrainer
from tore_train.argument_parser import ToreTrainArgumentParser

from tore_train.arguments import (
    ModelArguments,
    DataArguments,
    LoggingArguments,
)
from tore_train.model_utils import get_checkpoint, get_tokenizer
from tore_train.logging_utils import init_logging
from transformers import BatchFeature
from transformers.utils import is_datasets_available
from torch.utils.data import DataLoader

from tore_train.data_processors.packing.packing_index_processor import (
    PackingIndexProcessor,
    PackingIndexProcessorArguments,
)
from tore_train.data_processors.labels_masking_processor import (
    LabelsMaskingProcessor,
    LabelsMaskingProcessorArguments,
)
from tore_train.data_processors.vision.images_loader import ImagesLoader
from tore_train.data_processors.vision.videos_loader import VideosLoader

from tore_train.monkeypatch.qwen2_vl.cu_seqlen_attn import apply_qwen2vl_cu_seqlen_attn
from tore_train.monkeypatch.qwen2_vl.modeling_for_train import (
    apply_qwen2vl_modeling_for_train,
)

from qwen2vl_trainer import Qwen2VLTrainer

logger = logging.getLogger(__name__)

torch.cuda.set_per_process_memory_fraction(0.9)

@dataclass
class CustomModelArguments(ModelArguments):
    pass


@dataclass
class CustomDataArguments(DataArguments):
    assistant_bos: str = field(
        default="<|im_start|>assistant\n",
        metadata={"help": "The assistant start of sentence token."},
    )
    assistant_eos: str = field(
        default="<|im_end|>",
        metadata={"help": "The assistant end of sentence token."},
    )
    processor_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "The processor name or path."},
    )
    packing_algorithm: str = field(
        default="fast_best_fit_decreasing",
        metadata={"help": "The packing algorithm."},
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "The maximum sequence length."},
    )


def load_dataset(path, keep_in_memory=False, storage_options=None):
    if path.startswith("s3://"):
        return datasets.load_from_disk(
            path, keep_in_memory=keep_in_memory, storage_options=storage_options
        )

    # if os.pathconf
    # datasets.load_dataset()
    # return datasets.load_from_disk(path, keep_in_memory=keep_in_memory)


def main():
    parser = ToreTrainArgumentParser(
        (CustomModelArguments, CustomDataArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse()

    all_run_args = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args),
        **asdict(logging_args),
    }

    if "wandb" in training_args.report_to and os.environ.get("RANK", "0") == "0":
        import wandb

        wandb_all_args = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": asdict(training_args),
        }
        run = wandb.init(
            project=logging_args.wandb_project,
            name=(
                training_args.output_dir.split("/")[-1]
                if logging_args.wandb_run_name is None
                else logging_args.wandb_run_name
            ),
            group=logging_args.wandb_group,
            config=wandb_all_args,
        )
        run_id = run.id
        all_run_args["wandb_id"] = run_id

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    init_logging(model_args, data_args, training_args, logging_args)

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model args {model_args}")
    logger.info(f"Data args {data_args}")
    logger.info(f"Training/evaluation args {training_args}")
    logger.info(f"Logging args {logging_args}")

    #################
    # Find checkpoint
    #################
    last_checkpoint = get_checkpoint(training_args)

    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer
    ################
    vlm_processor = Qwen2VLProcessor.from_pretrained(data_args.processor_name_or_path)
    tokenizer = vlm_processor.tokenizer

    ###############
    # Load datasets
    ###############
    # TODO: support loading from huggingface datasets
    if data_args.train_dataset_path is not None:
        train_dataset = datasets.load_from_disk(data_args.train_dataset_path)
    else:
        raise ValueError("Train dataset path is not provided.")

    if data_args.eval_dataset_path is not None:
        eval_dataset = datasets.load_from_disk(data_args.eval_dataset_path)
    else:
        eval_dataset = None

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    # load the config
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )

    # Monkey patch
    model = apply_qwen2vl_modeling_for_train(model)

    # freeze model visual encoder
    # for name, param in model.visual.named_parameters():
    #     param.requires_grad = False

    # for name, param in model.visual.merger.named_parameters():
    #     param.requires_grad = False

    def mask_embedding_grad(grad):
        grad[:] = 0
        return grad

    model.model.embed_tokens.weight.register_hook(mask_embedding_grad)
    model.lm_head.weight.register_hook(mask_embedding_grad)

    ############################
    # Initialize the Data Loader
    ############################
    # vlm_processor = Qwen2VLProcessor.from_pretrained(data_args.processor_name_or_path)

    config = PackingIndexProcessorArguments(
        packing_algorithm=data_args.packing_algorithm,
        num_proc=64,
        max_seq_len=data_args.max_seq_length,
    )
    packing_processor = PackingIndexProcessor(config)

    config = LabelsMaskingProcessorArguments(
        assistant_bos=data_args.assistant_bos,
        assistant_eos=data_args.assistant_eos,
        tokenizer=tokenizer,
    )
    labels_masking_processor = LabelsMaskingProcessor(config)

    videos_loader = VideosLoader(
        endpoint_url=os.environ.get("TOGETHER_S3_ENDPOINT_URL")
    )

    # filter long sequences
    def filter_long_sequence(example, max_seq_len):
        if len(example["input_ids"]) > max_seq_len:
            return False
        else:
            return True

    def batch_filter_long_sequences(examples, max_seq_len):
        results = []
        for i in range(len(examples["input_ids"])):
            input_ids = examples["input_ids"][i]
            results.append(filter_long_sequence({"input_ids": input_ids}, max_seq_len))
        return results

    # filter long sequences
    print("Filtering train dataset")
    train_dataset = train_dataset.filter(
        batch_filter_long_sequences,
        fn_kwargs={"max_seq_len": data_args.max_seq_length},
        num_proc=64,
        batched=True,
        batch_size=1024,
    )
    # train_dataset.save_to_disk(
    #     data_args.train_dataset_path + "train_dataset_filtered.arrow"
    # )

    # Dynamic Packing based on the sequence length
    train_bins_dataset = packing_processor(train_dataset)
    print("Packing train dataset")
    train_bins_dataset = packing_processor(train_dataset)
    # train_bins_dataset.save_to_disk(
    #     data_args.train_dataset_path + "train_bins_dataset.arrow"
    # )

    # TODO: create a custom data collator for Qwen2VL
    def collate_fn(examples):
        all_input_ids = []
        all_labels = []
        all_position_ids = []
        all_attention_mask = []

        for example in examples:
            # process the example
            items = [train_dataset[idx] for idx in example["index_bins"]]
            list_input_ids = [item["input_ids"] for item in items]
            list_labels = [
                labels_masking_processor.get_labels(item)["labels"] for item in items
            ]
            list_position_ids = [
                torch.arange(len(list_input_ids[i])) for i in range(len(list_input_ids))
            ]

            list_input_ids = [
                torch.LongTensor(list_input_ids[i]) for i in range(len(list_input_ids))
            ]
            list_labels = [
                torch.LongTensor(list_labels[i]) for i in range(len(list_labels))
            ]

            input_ids = torch.cat(list_input_ids, dim=0)
            labels = torch.cat(list_labels, dim=0)
            position_ids = torch.cat(list_position_ids, dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_position_ids.append(position_ids)
            all_attention_mask.append(attention_mask)

        all_input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        all_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)
        all_position_ids = pad_sequence(
            all_position_ids, batch_first=True, padding_value=0
        )
        all_attention_mask = pad_sequence(
            all_attention_mask, batch_first=True, padding_value=False
        )

        # Get all images and process them
        all_video_paths = []
        for example in examples:
            items = [train_dataset[idx] for idx in example["index_bins"]]
            if "videos" not in items[0] or items[0]["videos"] is None:
                all_video_paths.extend([None] * len(items))
            else:
                list_video_paths = [video for item in items for video in item["videos"]]
                all_video_paths.extend(list_video_paths)

        if all_video_paths[0] is not None:
            all_videos = videos_loader.get_videos(all_video_paths)
            video_frames = []
            for video in all_videos:
                video_frames.append(video["frames"])
            processor_output = vlm_processor(
                images=None, videos=video_frames, text=vlm_processor.video_token * len(video_frames), return_tensors="pt"
            )

            pixel_values_videos = processor_output["pixel_values_videos"]
            video_grid_thw = processor_output["video_grid_thw"]
        else:
            pixel_values_videos = None
            video_grid_thw = None

        # Pad sequences to multiple of 128
        pad_len = 128 - (all_input_ids.shape[1] % 128)
        if pad_len < 128:
            pad_input_ids = torch.full(
                (all_input_ids.shape[0], pad_len),
                tokenizer.pad_token_id,
                dtype=all_input_ids.dtype,
                device=all_input_ids.device,
            )
            pad_labels = torch.full(
                (all_labels.shape[0], pad_len),
                -100,
                dtype=all_labels.dtype,
                device=all_labels.device,
            )
            pad_position_ids = torch.full(
                (all_position_ids.shape[0], pad_len),
                0,
                dtype=all_position_ids.dtype,
                device=all_position_ids.device,
            )
            pad_attention_mask = torch.full(
                (all_attention_mask.shape[0], pad_len),
                False,
                dtype=all_attention_mask.dtype,
                device=all_attention_mask.device,
            )

            all_input_ids = torch.cat([all_input_ids, pad_input_ids], dim=1)
            all_labels = torch.cat([all_labels, pad_labels], dim=1)
            all_position_ids = torch.cat([all_position_ids, pad_position_ids], dim=1)
            all_attention_mask = torch.cat(
                [all_attention_mask, pad_attention_mask], dim=1
            )

        if pixel_values_videos is not None:
            features = BatchFeature(
                data={
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "position_ids": all_position_ids,
                    "labels": all_labels,
                    "pixel_values_videos": pixel_values_videos,
                    "video_grid_thw": video_grid_thw,
                },
                tensor_type="pt",
            )
        else:
            features = BatchFeature(
                data={
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "position_ids": all_position_ids,
                    "labels": all_labels,
                },
                tensor_type="pt",
            )
        return features

    trainer = Qwen2VLTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        train_dataset=train_bins_dataset,
        # eval_dataset=eval_dataset,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Start training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
