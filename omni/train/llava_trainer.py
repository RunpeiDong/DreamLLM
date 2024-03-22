import os

import torch
from transformers import __version__
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_callback import DefaultFlowCallback, ProgressCallback
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from omni.train.trainer import Trainer
from omni.utils.fsdp_utils import save_dreamllm_fsdp_full_state_dict
from omni.utils.import_utils import is_accelerate_available, is_peft_available, is_safetensors_available
from omni.utils.loguru import logger

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


if is_peft_available():
    from peft import PeftModel


if is_safetensors_available():
    import safetensors.torch


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class LLaVATrainer(Trainer):
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.fsdp is not None or self.is_fsdp_enabled:
            state_dict = self.model.state_dict() if not self.is_fsdp_enabled else {}
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if self.is_fsdp_enabled:
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                # save_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir)
                save_dreamllm_fsdp_full_state_dict(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir)

        elif self.is_deepspeed_enabled:
            # this takes care of everything as long as we aren't under zero3
            if is_accelerate_available("<=", "0.20.3"):
                raise ValueError("Install Accelerate from main branch")
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use" " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save(self, output_dir: str | None = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)

        if self.fsdp is None and not self.is_fsdp_enabled:
            # save plugin modules
            for plugin_name, type in self.model.config.plugins_type.items():
                if type == "embedding":
                    getattr(self.model.get_decoder(), plugin_name).save_model(output_dir)
                elif type == "head":
                    getattr(self.model, plugin_name).save_model(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # NOTE: DreamLLM save model differently as all plugin modules are saved separately as bin files.
        #       Only the LLM is saved to checkpoint as safetensors or pytorch_model.bin.
        #       So the plugin modules such as vision encoders or SD should be loaded separately.
        logger.info("Resuming LLaVA from {}".format(resume_from_checkpoint))
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        logger.info(">>> Base language model resumed from {}".format(resume_from_checkpoint))

        logger.info("Resuming plugin modules from {}".format(resume_from_checkpoint))
        if model is None:
            model = self.model

        model.config.reset_plugins_init_kwargs(resume_from_checkpoint)
        model.init_plugin_modules()
        logger.info(">>> Plugin modules resumed from {}".format(resume_from_checkpoint))
