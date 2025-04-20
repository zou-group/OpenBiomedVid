from dataclasses import dataclass, field
from tore_train.arguments import ModelArguments

@dataclass
class SpeculatorArguments(ModelArguments):
    draft_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model config path. Don't set if you want to use the default config."
            )
        },
    )
    target_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    distillation_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "The weight for distillation loss. If 0, no distillation loss will be used."
            )
        },
    )
