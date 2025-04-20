# coding=utf-8
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
from typing import Dict
import os
from pathlib import Path

import torch
import shutil
from transformers import (
    TrainingArguments,
    AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer,
    AutoModelForCausalLM, GemmaForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftConfig
from typing import Union

from .arguments import ModelArguments, DataArguments


def get_peft_config(model_args: ModelArguments) -> Union[PeftConfig, None]:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def get_checkpoint(training_args: TrainingArguments) -> Union[Path, None]:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelArguments, data_args: DataArguments, auto_set_chat_template: bool = False
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
        if model_args.tokenizer_name_or_path is None
        else model_args.tokenizer_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def remove_optimizer_weights(save_dir):
    for checkpoint_dirs in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dirs)
        if os.path.isdir(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith('global_step'):
                    optimizer_dir = os.path.join(checkpoint_dir, file)
                    # remove the entire folder. This is used by deepspeed to store optimizer states
                    print('removing global_step', optimizer_dir)
                    shutil.rmtree(optimizer_dir)
                elif file.startswith("optimizer.pt"):
                    optimizer_file = os.path.join(checkpoint_dir, file)
                    print('removing optimizer', optimizer_file)
                    os.remove(optimizer_file)
    return


def fix_deepspeed_model_save(model_saved_path, dtype=torch.bfloat16):
    # for some model, we need to manually fix something when deepspeed save the model
    model = AutoModelForCausalLM.from_pretrained(model_saved_path, torch_dtype=dtype, device_map='cpu')
    if isinstance(model, GemmaForCausalLM):
        print("Fixing Gemma model")
        state_dict = model.state_dict()
        true_type = type(state_dict)
        fixed_state_dict = true_type({
            k: v.clone().cpu()
            for k, v in state_dict.items()
            if k != "lm_head.weight"
        })
        model.save_pretrained(
            model_saved_path, state_dict=fixed_state_dict, safe_serialization=True
        )
    return
