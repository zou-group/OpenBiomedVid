from typing import Any, Dict
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass, field
import warnings
import shutil

@dataclass
class SaveToDiskProcessorArguments:
    output_path: str = field(
        default=None,
        metadata={"help": "The path to save the dataset."},
    )
    max_shard_size: int = field(
        default="200MB",
        metadata={"help": "The maximum size of each shard."},
    )
    storage_options: Dict[str, Any] = field(
        default=None,
        metadata={"help": "The storage options."},
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite the output data"},
    )


class SaveToDiskProcessor:
    input_columns = []
    output_columns = []
    arguments = SaveToDiskProcessorArguments

    def __init__(self, config: SaveToDiskProcessorArguments):
        self.config = config
        self.output_path = config.output_path
        self.max_shard_size = config.max_shard_size
        self.storage_options = config.storage_options

    def __call__(self, dataset: Dataset) -> Dataset:
        return self.process(dataset)

    def process(self, dataset: Dataset) -> Dataset:
        if os.path.exists(self.output_path):
            if not self.config.overwrite:
                raise ValueError(f"Output location {self.output_path} already exists and overwrite is set to False.")
            else:
                # remove the data
                warnings.warn(f"Output location {self.output_path} already exists and will be overwritten.")
                shutil.rmtree(self.output_path)

        dataset.save_to_disk(
            self.output_path,
            max_shard_size=self.max_shard_size,
            storage_options=self.storage_options,
        )

        return dataset