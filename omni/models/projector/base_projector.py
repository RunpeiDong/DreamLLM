import torch
import torch.nn as nn

from omni.utils.loguru import logger
from omni.utils.misc import check_path_and_file


class BaseProjector(nn.Module):
    def load_model(self, model_name_or_path=None):
        if model_name_or_path is not None:
            if check_path_and_file(model_name_or_path, f"{self.save_model_name}_projector.bin"):
                logger.info(f"loading `BaseProjector` from {model_name_or_path}...")
                self.load_state_dict(torch.load(model_name_or_path, map_location="cpu"))
                return True
            # HACK: For compatibility
            if check_path_and_file(model_name_or_path, f"{self.save_model_name}_projector.pt"):
                logger.info(f"loading `BaseProjector` from {model_name_or_path}...")
                self.projector.load_state_dict(torch.load(model_name_or_path, map_location="cpu"))
                return True
        return False

    def forward(self, features) -> list:
        # NOTE return a list to be compatible with models using multiple paths
        pass

    @property
    def save_model_name(self):
        return self.args.save_model_name + "_projector"

    @property
    def dtype(self):
        return self.projector.dtype

    @property
    def device(self):
        return self.projector.device
