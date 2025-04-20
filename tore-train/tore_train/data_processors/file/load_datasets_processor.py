from typing import Any, Callable, List, Dict
import os
import collections
import tqdm
from dataclasses import dataclass, field
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from datasets.exceptions import DatasetGenerationError


@dataclass
class LoadDatasetsProcessorArguments:
    datasets: dict = field(
        default=None,
        metadata={"help": "The dataset name to be loaded"}
    )
    dataset_random_seed: int = field(
        default=42,
        metadata={"help": "The random seed to use for shuffling."},
    )
    dataset_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the dataset."},
    )
    dataset_columns_to_keep: List[str] = field(
        default=None,
        metadata={"help": "The columns to keep."},
    )
    dataset_dedup_key: str = field(
        default=None,
        metadata={"help": "The key to deduplicate the dataset."},
    )
    dataset_num_proc: int = field(
        default=128,
        metadata={"help": "The number of processes to use."},
    )
    dataset_storage_options: dict = field(
        default=None,
        metadata={"help": "The storage options to use."},
    )


# TODO: Add support for S3 support

class LoadDatasetsProcessor:
    input_columns = ["path", "subset", "split", "ratio"]
    output_columns = ["*"]
    arguments = LoadDatasetsProcessorArguments

    def __init__(self, config: LoadDatasetsProcessorArguments, process_fn: Callable = None):
        self.config = config
        self.shuffle = config.dataset_shuffle
        self.columns_to_keep = config.dataset_columns_to_keep
        self.dedup_key = config.dataset_dedup_key
        self.random_seed = config.dataset_random_seed
        self.num_proc = config.dataset_num_proc
        self.process_fn = process_fn
        self.storage_options = config.dataset_storage_options

    def __call__(self, datasets: Dataset | List[Dict]) -> Dataset:
        return self.process(datasets)

    def process(self, dataset_mix: List[Dict]) -> Dataset:
        dataset_mix_paths = [dataset["path"] for dataset in dataset_mix]
        
        # show that the following datasets are being loaded
        print(f"Loading datasets: {dataset_mix_paths}...")

        all_hf_datasets = []
        fracs = []
        dataset_splits = []
        dataset_subsets = []

        for idx, dataset_path in enumerate(
            tqdm.tqdm(dataset_mix_paths, desc="Loading datasets")
        ):
            # Check if splits are provided for each dataset
            dataset_split = dataset_mix[idx].get("split", "train")
            dataset_subset = dataset_mix[idx].get("subset", None)
            frac = dataset_mix[idx].get("ratio", 1.0)
            dataset_splits.append(dataset_split)
            dataset_subsets.append(dataset_subset)
            fracs.append(frac)

            print(f"Loading dataset {dataset_path} with split {dataset_split}, subset {dataset_subset}, and ratio {frac}")

            # If from a local path, load from disk
            if os.path.exists(dataset_path):
                # if json
                if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
                    dataset = load_dataset("json", data_files=dataset_path, num_proc=self.num_proc)
                elif dataset_path.endswith(".csv"):
                    dataset = load_dataset("csv", data_files=dataset_path, num_proc=self.num_proc)
                else:
                    # local dataset
                    dataset = load_from_disk(dataset_path)

                if dataset_split is not None:
                    dataset = dataset[dataset_split]
            else:
                # Try loading from Hugging Face Repo
                try:
                    dataset = load_dataset(
                        dataset_path,
                        dataset_subset,
                        split=dataset_split,
                        storage_options=self.storage_options,
                    )
                except Exception as e:
                    raise ValueError(f"Dataset {dataset_path} not found on HuggingFace Hub or locally")

            # Remove redundant columns to avoid schema conflicts on load
            if self.columns_to_keep is not None:
                dataset = dataset.remove_columns(
                    [
                        col
                        for col in dataset.column_names
                        if col not in self.columns_to_keep
                    ]
                )

            if (
                self.dedup_key is not None
                and self.dedup_key not in dataset.column_names
            ):
                raise ValueError(f"Key {self.dedup_key} not found in dataset columns.")

            dataset = dataset.add_column(
                "dataset_mix_source", [dataset_path] * len(dataset)
            )

            all_hf_datasets.append(dataset)

        if any(frac < 0 for frac in fracs):
            raise ValueError("Dataset fractions cannot be negative.")

        if self.dedup_key is not None:
            all_hf_datasets = concatenate_datasets(all_hf_datasets)

            # remove duplicates
            all_hf_datasets = all_hf_datasets.map(
                lambda x: {"temp_text": str(x[self.dedup_key])}, num_proc=self.num_proc
            )

            # load everything into memory
            texts = all_hf_datasets["temp_text"]

            uniques = collections.defaultdict(list)
            not_dups = []

            # hash all texts and find duplicates
            for i in tqdm.tqdm(range(len(texts)), desc="Finding duplicates"):
                text = texts[i]
                if hash(text) in uniques:
                    uniques[hash(text)].append(i)
                    not_dups.append(False)
                else:
                    uniques[hash(text)].append(i)
                    not_dups.append(True)

            # remove duplicates based on hash
            all_hf_datasets = all_hf_datasets.filter(
                lambda example, idx: not_dups[idx],
                with_indices=True,
                num_proc=self.num_proc,
            )
            all_hf_datasets = all_hf_datasets.remove_columns(["temp_text"])

            # print the ratio of duplicates
            print(f"Ratio of duplicates: {1 - sum(not_dups) / len(texts)}")

            # split all_raw_data back to different datasets based on source
            datasets_map = {}
            for dataset_path in tqdm.tqdm(
                dataset_mix_paths, desc="Splitting datasets back to original sources"
            ):
                datasets_map[dataset_path] = all_hf_datasets.filter(
                    lambda example: example["dataset_mix_source"] == dataset_path,
                    num_proc=self.num_proc,
                )

            # transform back to list
            all_hf_datasets = list(datasets_map.values())

        if self.process_fn is not None:
            all_hf_datasets = [
                self.process_fn(dataset) for dataset in all_hf_datasets
            ]

        # Combine all datasets based on the mixer fractions
        final_datasets = []
        for idx, dataset_path in enumerate(dataset_mix_paths):
            dataset = all_hf_datasets[idx]
            frac = fracs[idx]
            # Repeat dataset frac int times
            repeat = int(frac)
            for _ in range(repeat):
                if self.shuffle:
                    dataset = dataset.shuffle(seed=self.random_seed)
                final_datasets.append(dataset)

            # Add the remaining float fraction
            frac = frac % 1
            if frac > 0:
                if self.shuffle:
                    dataset = dataset.shuffle(seed=self.random_seed)
                train_subset = dataset.select(range(int(frac * len(dataset))))
                final_datasets.append(train_subset)

        if self.shuffle:
            final_datasets = concatenate_datasets(final_datasets).shuffle(
                seed=self.random_seed
            )
        else:
            final_datasets = concatenate_datasets(final_datasets)

        if len(final_datasets) == 0:
            raise ValueError("No datasets found.")

        return final_datasets