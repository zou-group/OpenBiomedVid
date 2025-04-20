import logging
import torch
import sys
import transformers
import datasets
from dataclasses import asdict
import torch.distributed as dist
import yaml
import os

logger = logging.getLogger(__name__)


def init_logging(model_args, data_args, training_args, logging_args):
    all_run_args = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args),
        **asdict(logging_args),
    }

    ### Init wandb ###
    # make sure only rank 0 will init wandb
    if (
        "wandb" in training_args.report_to
        and training_args.local_rank in [-1, 0]
        and int(os.environ.get("RANK", 0)) == 0
    ):
        import wandb

        wandb_all_args = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": asdict(training_args),
        }

        # Initialize wandb run
        if logging_args.wandb_id is not None:
            run = wandb.init(
                project=logging_args.wandb_project,
                name=logging_args.wandb_run_name,
                group=logging_args.wandb_group,
                config=wandb_all_args,
                id=logging_args.wandb_id,
            )
        else:
            run = wandb.init(
                project=logging_args.wandb_project,
                name=logging_args.wandb_run_name,
                group=logging_args.wandb_group,
                config=wandb_all_args,
            )
        run_id = run.id
        all_run_args["wandb_id"] = run_id

        # save all_run_args to a yaml file
        os.makedirs(training_args.output_dir, exist_ok=True)
        if os.path.exists(os.path.join(training_args.output_dir, "run_args.yaml")):
            if training_args.overwrite_output_dir:
                with open(
                    os.path.join(training_args.output_dir, "run_args.yaml"), "w"
                ) as f:
                    yaml.dump(all_run_args, f)
            else:
                raise ValueError(
                    f"Output directory {training_args.output_dir} already exists and is not empty."
                )
        else:
            with open(
                os.path.join(training_args.output_dir, "run_args.yaml"), "w"
            ) as f:
                yaml.dump(all_run_args, f)

    ### Init logger ###
    init_logger(is_main=is_main_rank(), log_level=logging.INFO)


def init_logger(
    is_main=True, log_level=logging.ERROR, is_distributed=False, filename=None
):
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers.tokenization_utils.logger.setLevel(log_level)
    transformers.tokenization_utils_base.logger.setLevel(log_level)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_rank():
    return get_rank() == 0
