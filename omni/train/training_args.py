from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments as TF_TrainingArguments

from omni.utils.loguru import logger


@dataclass
class TrainingArguments(TF_TrainingArguments):
    """
    Inherited from the transformers.TrainingArguments Class
    Some additional parameters are not shown in this class, please refer to following url for a full parameter list.
    https://github.com/huggingface/transformers/blob/701298d2d3d5c7bde45e71cce12736098e3f05ef/src/transformers/training_args.py#L162
    """

    disable_tqdm: bool | None = field(default=True, metadata={"help": "Whether or not to disable the tqdm progress bars."})
    # Do not touch this type annotation or it will stop working in CLI
    fsdp_config: str | dict | None = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    val_steps: int | None = field(default=None, metadata={"help": ("Run an validation every X steps.")})
    validation_data: Any | list[Any] | None = field(
        default=None, metadata={"help": "Data that is used during validation to verify that the model is learning."}
    )
    report_to: str | list[str] | None = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    run_project: str | None = field(
        default=None, metadata={"help": "An optional descriptor for the run project. Notably used for wandb logging."}
    )
    scale_lr: bool = field(
        default=False,
        metadata={"help": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."},
    )
    unfreeze_vit: bool = field(default=False, metadata={"help": "Whether unfreeze vision encoder."})
    unfreeze_llm: bool = field(default=False, metadata={"help": "Whether unfreeze llm."})
    vit_llrd: bool = field(default=False, metadata={"help": "Whether use llrd for vision encoder."})
    llm_llrd: bool = field(default=False, metadata={"help": "Whether use llrd for LLM."})
    use_ema: bool = field(default=False, metadata={"help": "Whether to use EMA model."})

    def __post_init__(self):
        super().__post_init__()

        # val_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.val_steps is None or self.val_steps == 0:
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `val_steps` to {self.logging_steps}")
                self.val_steps = self.logging_steps
            else:
                raise ValueError("`val_steps` has to be defined and non-zero.")

        if self.validation_data is not None:
            if not isinstance(self.validation_data, list):
                self.validation_data = [self.validation_data]

        assert self.validation_data is None or isinstance(
            self.validation_data, list
        ), f"`validation_data` must be `None` or `list`, got {type(self.validation_data)}"
