from typing import Any
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass, field
import json

@dataclass
class TextReplaceProcessorArguments:
    replace_map: dict = field(
        default=None,
        metadata={"help": "The replace map to use."},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "The number of processes to use."},
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."},
    )

class TextReplaceProcessor:
    input_columns = ["text"]
    output_columns = ["text"]
    arguments = TextReplaceProcessorArguments

    def __init__(self, config: TextReplaceProcessorArguments):
        self.config = config
    
    def precheck(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Invalid dataset type: {type(dataset)}")
        if "text" not in dataset.column_names:
            raise ValueError(f"Invalid dataset: `text` field not found")

    def process(self, example: dict) -> dict:
        text = example["text"]
        for old, new in self.config.replace_map.items():
            text = text.replace(old, new)
        example["text"] = text
        return example

    def __call__(self, dataset: Dataset) -> Dataset:
        self.precheck(dataset)
        dataset = dataset.map(
            self.process,
            num_proc=self.config.num_proc,
            keep_in_memory=self.config.keep_in_memory,
        )
        return dataset
