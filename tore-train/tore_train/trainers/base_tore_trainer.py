import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from accelerate.state import PartialState
from datasets import Dataset
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.utils import is_datasets_available
import datasets

from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


class ToreTrainer(Trainer):
    """
    This is a base Trainer class for Tore.
    A wrapper around the `transformers.Trainer` class to store metrics during training and evaluation.
    """

    def __init__(self, *largs, **kwargs):
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.train_eval = "train"
        self.start_time = time.perf_counter()
        super().__init__(*largs, **kwargs)

    def store_metrics(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self._stored_metrics[self.train_eval][key].append(value)

    @wraps(Trainer.log)
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`float`, optional):
                The start time of the training.
        """
        # logs based on self.train_eval
        train_eval = self.train_eval
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.prediction_step)
    def prediction_step(self, *args, **kwargs):
        self.train_eval = "eval"
        return super().prediction_step(*args, **kwargs)

    # @wraps(Trainer.training_step)
    # def training_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     num_items_in_batch: Optional[int] = None,
    # ) -> torch.Tensor:
    #     self.train_eval = "train"
    #     if self.is_local_process_zero():
    #         # caclulate throughput
    #         if "input_ids" in inputs:
    #             input_ids = inputs["input_ids"]
    #             seq_len = input_ids.shape[1]
    #             batch_size = input_ids.shape[0]
    #             throughput = batch_size * seq_len * self.args.world_size
    #             current_time = time.perf_counter()
    #             elapsed_time = current_time - self.start_time
    #             throughput = throughput / elapsed_time
    #             self.store_metrics({"train_tps": throughput})

    #             self.start_time = time.perf_counter()
    #     tr_loss = super().training_step(model, inputs, num_items_in_batch)
    #     return tr_loss


    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        self.train_eval = "train"

        # measure throughput
        if self.is_local_process_zero():
            # caclulate throughput
            if "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                seq_len = input_ids.shape[1]
                batch_size = input_ids.shape[0]
                throughput = batch_size * seq_len * self.args.world_size
                current_time = time.perf_counter()
                elapsed_time = current_time - self.start_time
                throughput = throughput / elapsed_time
                self.store_metrics({"train_tps": throughput})

                self.start_time = time.perf_counter()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)

        # For deepspeed, we need to normalize the loss for reporting
        if self.is_deepspeed_enabled:
            loss = loss / self.args.gradient_accumulation_steps

        return loss.detach()


    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: Optional[str] = None
    ):
        return dataset
