from types import SimpleNamespace

from omni.utils.loguru import logger
from omni.models.projector.mlp_projector import LinearProjector, MLPProjector
from omni.models.projector.conv_projector import ConvProjector
from omni.models.projector.sam_projector import SAMProjector


def build_projector(projector_cfg, in_hidden_size, out_hidden_size, bias=None):
    projector_cfg = SimpleNamespace(**projector_cfg)
    projector = getattr(projector_cfg, 'projector', None)
    logger.info(f"Building projector ({projector_cfg.save_model_name}): {projector}")
    if projector == "linear":
        return LinearProjector(args=projector_cfg, in_hidden_size=in_hidden_size, out_hidden_size=out_hidden_size, bias=bias)
    if projector == "mlp":
        return MLPProjector(args=projector_cfg, in_hidden_size=in_hidden_size, out_hidden_size=out_hidden_size, bias=bias)
    elif projector == "conv":
        return ConvProjector(args=projector_cfg, in_hidden_size=in_hidden_size, out_hidden_size=out_hidden_size)
    elif projector == "sam":
        return SAMProjector(args=projector_cfg, in_hidden_size=in_hidden_size, out_hidden_size=out_hidden_size)

    raise ValueError(f"Unknown projector: {projector} (supported: linear, mlp, conv, sam)")
