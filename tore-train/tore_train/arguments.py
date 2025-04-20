# coding=utf-8
# This file is copied from alignment-handbook/configs.py
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)

@dataclass
class ModelArguments:
    """
    Arguments for loading and configuring the model.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the dataset."},
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the dataset."},
    )
    truncation_side: Optional[str] = field(
        default=None,
        metadata={"help": "The side of truncation."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template."},
    )


@dataclass
class LoggingArguments:
    """
    Arguments pertaining to remote logging
    """
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The wandb project to use for logging.")},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The wandb group to use for logging.")},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The wandb name to use for logging.")},
    )
    wandb_id: Optional[str] = field(
        default=None,
        metadata={"help": ("The wandb id to use for logging.")},
    )
