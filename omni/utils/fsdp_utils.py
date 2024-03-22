import os
from collections import OrderedDict

import torch
from accelerate.utils.constants import FSDP_PYTORCH_VERSION, MODEL_NAME

from omni.utils.import_utils import is_torch_available, is_torch_distributed_available
from omni.utils.loguru import logger

if is_torch_available(">=", FSDP_PYTORCH_VERSION) and is_torch_distributed_available():
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


class FSDPMixin:
    def fsdp_ignored_modules(self) -> list:
        return []


def save_dreamllm_fsdp_full_state_dict(fsdp_plugin, accelerator, model, output_dir):
    assert fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT, "Only FULL_STATE_DICT is supported now."

    os.makedirs(output_dir, exist_ok=True)

    # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
    # so, only enable it when num_processes>1
    is_multi_process = accelerator.num_processes > 1
    fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
    fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    with FSDP.state_dict_type(
        model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
    ):
        state_dict = model.state_dict()

        weights_name_dict = {}
        for plugin_name, type in model.config.plugins_type.items():
            if type == "embedding":
                plugin_model = getattr(model.get_decoder(), plugin_name)
                prefix = f"model.{plugin_name}."
            elif type == "head":
                plugin_model = getattr(model, plugin_name)
                prefix = f"{plugin_name}."
            weights_name_dict[plugin_model.save_model_name] = OrderedDict(
                (key[len(prefix) :], value) for key, value in state_dict.items() if key.startswith(prefix)
            )

        if accelerator.process_index == 0:
            output_model_file = os.path.join(output_dir, f"{MODEL_NAME}.bin")
            logger.info(f"Saving model to {output_model_file}")
            torch.save(state_dict, output_model_file)
            logger.info(f"Model saved to {output_model_file}")

            for _model_name, _state_dict in weights_name_dict.items():
                output_model_file = os.path.join(output_dir, f"{_model_name}.bin")
                logger.info(f"Saving model to {output_model_file}")
                torch.save(_state_dict, output_model_file)
                logger.info(f"Model saved to {output_model_file}")
