from typing import Any, List
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass, field

from tore_train.data_processors.packing.fast_fast_best_fit_decreasing import fast_best_fit_decreasing



@dataclass
class PackingIndexProcessorArguments:
    packing_algorithm: str = field(
        default="none",
        metadata={"help": "The algorithm to use for packing."},
    )
    packing_random_seed: int = field(
        default=42,
        metadata={"help": "The random seed to use for packing."},
    )
    max_seq_len: int = field(
        default=8192,
        metadata={"help": "The maximum sequence length."},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "The number of processes to use."},
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."},
    )

class PackingIndexProcessor:
    input_columns = ["input_ids"]
    output_columns = ["index_bins"]
    arguments = PackingIndexProcessorArguments

    def __init__(self, config: PackingIndexProcessorArguments):
        self.config = config
        self.num_proc = config.num_proc
        self.random_seed = config.packing_random_seed
        self.max_seq_len = config.max_seq_len

        self.packing_algorithm = config.packing_algorithm


    def get_lengths(self, dataset: Dataset) -> List[int]:
        def get_length(example):
            length = len(example["input_ids"])
            if length > self.max_seq_len:
                raise ValueError(f"Sequence length {length} is greater than the maximum sequence length {self.max_seq_len}")
            return {"length": length}

        temp_dataset = dataset.map(
            get_length,
            num_proc=self.num_proc,
        )
        return list(temp_dataset["length"])

    def next_fit(self, lengths) -> Dataset:
        # randomize the order of the lengths
        rng = np.random.default_rng(self.random_seed)
        indexed_lengths = [(i, length) for i, length in enumerate(lengths)]
        rng.shuffle(indexed_lengths)

        bins = [[]]
        current_bin_length = 0
        for index, length in indexed_lengths:
            if current_bin_length + length > self.max_seq_len:
                bins.append([])
                current_bin_length = 0
            bins[-1].append(index)
            current_bin_length += length

        # Convert back to original indices
        original_bins = [[] for _ in range(len(bins))]
        for bin_index, item_indices in enumerate(bins):
            for item_index in item_indices:
                original_bins[bin_index].append(indexed_lengths[item_index][0])
        bins = original_bins

        # shuffle bins
        rng = np.random.default_rng(self.random_seed)
        for bin in bins:
            rng.shuffle(bin)

        return bins

    def fast_ffd_packing(self, lengths: List[int]) -> List[List[int]]:
        rng = np.random.default_rng(self.random_seed)
        indexed_lengths = [(i, length) for i, length in enumerate(lengths)]
        rng.shuffle(indexed_lengths)

        lengths = [indexed_lengths[i][1] for i in range(len(indexed_lengths))]
        
        bins = fast_best_fit_decreasing(lengths, self.max_seq_len, progress_bar=True)

        # shuffle bins
        # Convert back to original indices
        original_bins = [[] for _ in range(len(bins))]
        for bin_index, item_indices in enumerate(bins):
            for item_index in item_indices:
                original_bins[bin_index].append(indexed_lengths[item_index][0])
        bins = original_bins
    
        rng = np.random.default_rng(self.random_seed)
        for bin in bins:
            rng.shuffle(bin)

        return bins

    def __call__(self, dataset: Dataset) -> Dataset:
        if "input_ids" not in dataset.column_names:
            raise ValueError("input_ids column is required")
        
        return self.process(dataset)

    def process(self, dataset: Dataset) -> Dataset:
        lengths = self.get_lengths(dataset)

        if self.packing_algorithm == "fast_best_fit_decreasing":
            bins = self.fast_ffd_packing(lengths)
        elif self.packing_algorithm == "next_fit":
            bins = self.next_fit(lengths)
        else:
            raise ValueError(f"packing algorithm {self.packing_algorithm} is not supported")

        bins_dataset = Dataset.from_dict({"index_bins": bins})

        return bins_dataset