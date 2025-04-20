from typing import Dict, List, Optional, Union, Any
from copy import deepcopy
import random
import wandb
import warnings
from accelerate.utils import is_deepspeed_available, tqdm
from accelerate.utils import gather_object
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from transformers import Trainer, PreTrainedModel


from tore_train.trainer.base_tore_trainer import ToreTrainer

from .metrics import get_acceptance_rate
from .speculator_arguments import SpeculatorArguments


if is_deepspeed_available():
    import deepspeed

class SpeculatorTrainer(ToreTrainer):
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        speculator_args: SpeculatorArguments,
        *largs,
        **kwargs,
    ):
        self.draft_model = draft_model

        self.target_model = target_model
        self.target_model.eval()

        # disable the gradient for target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        super().__init__(*largs, model=draft_model, **kwargs)

        self.distillation_loss_weight = speculator_args.distillation_loss_weight

        # prepare the target model for deepspeed
        if self.is_deepspeed_enabled:
            self.target_model = self._prepare_deepspeed(self.target_model)
        else:
            self.target_model = self.accelerator.prepare_model(
                self.target_model, evaluation_mode=True
            )

    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        draft_model = model
        target_model = self.target_model

        draft_outputs = draft_model(**inputs)

        # if distillation loss weight is 0, we only return the draft loss
        if self.distillation_loss_weight == 0.0:
            return draft_outputs.loss

        target_outputs = target_model(**inputs)
        target_loss = target_outputs.loss.item()
        draft_loss = draft_outputs.loss

        target_logits = target_outputs.logits
        draft_logits = draft_outputs.logits

        # Calculate the KL divergence between the target and draft logits
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        kl_div = kl_loss(
            torch.log_softmax(draft_logits, dim=-1),
            torch.softmax(target_logits, dim=-1),
        )

        loss = (
            1 - self.distillation_loss_weight
        ) * draft_loss + self.distillation_loss_weight * kl_div

        metrics = {
            "target_lm_loss": target_loss,
            "draft_lm_loss": draft_loss.item(),
            "distill_kl_div": kl_div.item(),
        }

        self.store_metrics(metrics)

        return loss

    # caculate acceptance rate
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        draft_model = model
        target_model = self.target_model

        draft_model.eval()
        target_model.eval()

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        labels = inputs["labels"]

        with torch.no_grad():
            draft_output = draft_model(**inputs)
            # assume the target model has the same input signature as the draft model
            target_output = target_model(**inputs)

            target_logits = target_output.logits
            # only keep non-ignored tokens
            target_valid_logits_len = (labels != -100).sum(dim=-1)
            target_logits = target_logits[labels != -100]

            draft_output = draft_model(**inputs)
            draft_logits = draft_output.logits
            # only keep non-ignored tokens
            draft_valid_logits_len = (labels != -100).sum(dim=-1)
            draft_logits = draft_logits[labels != -100]

            target_logits = target_logits.unsqueeze(0)
            draft_logits = draft_logits.unsqueeze(0)

            accept_rates = get_acceptance_rate(
                target_logits, draft_logits, temp=1.0, top_p=1.0
            )

            # restore 1D shape
            accept_rates = accept_rates.squeeze(0)

            # Use torch.split to divide accept_rates into segments for batch_size
            accept_rates_segments = torch.split(
                accept_rates, draft_valid_logits_len.tolist()
            )

            # Calculate mean for each segment in the batch
            accept_rates = torch.cat(
                [segment.mean().unsqueeze(0) for segment in accept_rates_segments]
            )
            self.local_accept_rates.extend(accept_rates.tolist())

        return (draft_output.loss.detach(), None, None)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        self.local_accept_rates = []

        # Base evaluation
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        # calculate the mean of all accept rates
        self.accelerator.wait_for_everyone()
        # Gather results from all processes
        all_accept_rates = gather_object(self.local_accept_rates)
        mean_accept_rate = sum(all_accept_rates) / len(all_accept_rates)

        self.store_metrics({f"{metric_key_prefix}_accept_rate": mean_accept_rate})

        return output
