from typing import Any
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass, field
import json

def maybe_insert_system_message(messages, tokenizer, prob=1.0):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if prob == 1.0 or np.random.rand() < prob:
        # since it is maybe, adds a prob not always insert
        messages.insert(
            0, {"role": "system", "content": "You are a helpful assistant."}
        )


def apply_chat_template(example, tokenizer, auto_insert_empty_system_msg=True):
    messages = example["messages"]

    if isinstance(messages, str):
        messages = json.loads(messages)

    assert isinstance(messages, list), f"Invalid messages type: {type(messages)}"

    # We add an empty system message if there is none
    if auto_insert_empty_system_msg:
        maybe_insert_system_message(messages, tokenizer)

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


@dataclass
class ChatProcessorArguments:
    tokenizer: str = field(
        default=None,
        metadata={"help": "The tokenizer to use."},
    )
    chat_template: str = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={"help": "Whether to automatically insert an empty system message."},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "The number of processes to use."},
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."},
    )


class ChatProcessor:
    # TODO: refactor to support different messages formats and output text only
    # define the input and output columns
    # for future, we can combine multiple processors into one pipeline
    input_columns = ["*"]
    output_columns = ["text"]
    arguments = ChatProcessorArguments

    def __init__(self, config: ChatProcessorArguments):
        self.config = config

        if isinstance(self.config.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        elif isinstance(self.config.tokenizer, AutoTokenizer):
            self.tokenizer = self.config.tokenizer
        else:
            raise ValueError(f"Invalid tokenizer type: {type(self.config.tokenizer)}")

        if self.config.chat_template is not None:
            self.tokenizer.chat_template = self.config.chat_template

    def precheck(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        if "messages" not in dataset.column_names:
            # TODO: add more types check
            raise ValueError(f"Invalid dataset: `messages` field not found")

    def postcheck(self, dataset: Dataset):
        if "input_ids" not in dataset.column_names:
            raise ValueError(f"Invalid dataset: `input_ids` field not found")

    def __call__(self, dataset: Dataset):
        return self.process(dataset)

    def process(self, dataset: Dataset):
        self.precheck(dataset)

        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
            num_proc=self.config.num_proc,
            keep_in_memory=self.config.keep_in_memory,
        )

        return dataset