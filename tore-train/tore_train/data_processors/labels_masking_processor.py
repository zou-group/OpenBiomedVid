import copy
import torch
import numpy as np
from typing import Union, List
import warnings
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from datasets import Dataset


@dataclass
class LabelsMaskingProcessorArguments:
    tokenizer: Union[str, AutoTokenizer] = field(
        default=None,
        metadata={"help": "The tokenizer to use."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code."},
    )
    label_masking: bool = field(
        default=True,
        metadata={"help": "Whether to mask the labels."},
    )
    prob_no_label_mask: float = field(
        default=0.0,
        metadata={"help": "The probability of no mask."},
    )
    assistant_bos: Union[str, List[int]] = field(
        default="",
        metadata={"help": "The assistant bos token."},
    )
    assistant_eos: Union[str, List[int]] = field(
        default="",
        metadata={"help": "The assistant eos token."},
    )
    ignore_index: int = field(
        default=-100,
        metadata={"help": "The ignore index."},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "The number of processes to use."},
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."},
    )


class LabelsMaskingProcessor:
    input_columns = ["input_ids"]
    output_columns = ["labels"]
    arguments = LabelsMaskingProcessorArguments

    def __init__(self, config: LabelsMaskingProcessorArguments):
        self.config = config
        self.num_proc = config.num_proc
        self.keep_in_memory = config.keep_in_memory
        self.prob_no_mask = config.prob_no_label_mask

        if isinstance(self.config.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer, trust_remote_code=self.config.trust_remote_code)
        else:
            self.tokenizer = self.config.tokenizer

        self.assistant_bos = config.assistant_bos
        if isinstance(self.assistant_bos, str):
            # The user provides a string, must tokenize
            self.assistant_bos_token_ids = self.tokenizer.encode(
                self.assistant_bos, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.assistant_bos_token_ids = self.assistant_bos

        self.assistant_eos = config.assistant_eos
        if isinstance(self.assistant_eos, str):
            # The user provides a string, must tokenize
            self.assistant_eos_token_ids = self.tokenizer.encode(
                self.assistant_eos, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.assistant_eos_token_ids = self.assistant_eos

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = (
            config.ignore_index if hasattr(config, "ignore_index") else -100
        )

    def __call__(self, dataset: Dataset) -> Dataset:
        return self.process(dataset)

    def get_labels(self, example):
        if "labels" not in example:
            example["labels"] = copy.deepcopy(example["input_ids"])

        if self.prob_no_mask > 0 and np.random.rand() < self.prob_no_mask:
            return example

        labels = example["labels"]

        if isinstance(labels, list):
            labels = torch.tensor(labels)

        labels = labels.clone()

        assistant_bos_idxs = []
        eos_idxs = []

        for assistant_idx in np.where(labels == self.assistant_bos_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                self.assistant_bos_token_ids
                == labels[
                    assistant_idx : assistant_idx + len(self.assistant_bos_token_ids)
                ].tolist()
            ):
                assistant_bos_idxs.append(
                    assistant_idx + len(self.assistant_bos_token_ids)
                )

        if len(assistant_bos_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{self.assistant_bos}` in the "
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            labels[:] = self.ignore_index

        for eos_idx in np.where(labels == self.assistant_eos_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                self.assistant_eos_token_ids
                == labels[
                    eos_idx : eos_idx + len(self.assistant_eos_token_ids)
                ].tolist()
            ):
                eos_idxs.append(eos_idx + len(self.assistant_eos_token_ids))

        # create a list of tuples with the start and end indices
        indices = []
        for bos_idx in assistant_bos_idxs:
            indices.append((bos_idx, "start"))
        for eos_idx in eos_idxs:
            indices.append((eos_idx, "end"))

        # sort the indices by the start index
        # this is to find the start, end intervals
        indices = sorted(indices, key=lambda x: x[0])

        # find start, end intervals and set the labels outside to -100
        idx = 0
        prev_end = 0
        while idx < len(indices):
            if indices[idx][1] == "start":
                start = indices[idx][0]
                labels[prev_end:start] = self.ignore_index

                if idx + 1 < len(indices) and indices[idx + 1][1] == "end":
                    end = indices[idx + 1][0]
                    prev_end = end

            idx += 1

        # for the remaining labels, set to -100
        labels[prev_end:] = self.ignore_index

        # return the example with the new labels
        example["labels"] = labels
        return example

    def filter_element(self, element):
        # if all labels are -100, we filter out the example
        labels = element["labels"]
        if all(label == self.ignore_index for label in labels):
            return False
        return True

    def process(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            self.get_labels,
            num_proc=self.num_proc,
            keep_in_memory=self.keep_in_memory,
        )
        # filter out the examples that have all labels as -100
        dataset = dataset.filter(
            self.filter_element,
            num_proc=self.num_proc,
            keep_in_memory=self.keep_in_memory,
        )
        return dataset