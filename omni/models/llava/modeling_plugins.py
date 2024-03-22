import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.controlnet import MultiControlNetModel
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from omni.models.projector.builder import build_projector
from omni.utils.fsdp_utils import FSDPMixin
from omni.utils.import_utils import is_xformers_available
from omni.utils.loguru import logger
from omni.utils.misc import check_path_and_file
from omni.utils.modeling_utils import get_model_device, get_model_dtype
from omni.utils.torch_utils import is_compiled_module, randn_tensor

PluginType = Literal["embedding", "head"]
PipelineImageType = (
    PIL.Image.Image | np.ndarray | torch.FloatTensor | list[PIL.Image.Image] | list[torch.FloatTensor] | list[np.ndarray]
)


class PluginBase(ABC, nn.Module, FSDPMixin):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = None

    def _init_weights(self, module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)

    @property
    def device(self):
        return get_model_device(self)

    @property
    def dtype(self):
        return get_model_dtype(self)

    @property
    @abstractmethod
    def processor(self):
        """
        In `MultimodalEmbedding`, it is similar to text tokenizer, it processes signals into a form that can be understood by `MultimodalEmbedding`.
        In `MultimodalHead`, it will process signals into a form that can be **trained** by `MultimodalHead`.
        """
        pass

    @property
    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def save_model(self, output_dir: str):
        pass

    @abstractmethod
    def load_model(self, output_dir: str):
        pass

    @abstractmethod
    def forward(self):
        pass


class MultimodalEmbedding(PluginBase):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = "embedding"

    @property
    @abstractmethod
    def embed_len(self):
        """
        The length a signal after being processed by MultimodalEmbedding.
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self):
        """
        The dimension of the embedding.
        """
        pass


class MultimodalHead(PluginBase):
    initializer_range: float = 0.02
    plugin_type: PluginType | None = "head"

    @abstractmethod
    @torch.no_grad()
    def pipeline(self):
        pass


class CLIPVisionEmbedding(MultimodalEmbedding):
    def __init__(
        self,
        clip_vision_model_name_or_path: str,
        projector_type: str = "linear",
        projector_depth: int = 1,
        projector_name_or_path: str = None,
        pretrained_model_name_or_path: str | None = None,
        use_additional_post_layernorm: bool = False,
        select_layer: int = -2,
        embed_hidden_size: int = 4096,
        freeze_clip_vision_model: bool = True,
        freeze_embedding_layers: bool = True,
        freeze_projector: bool = False,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.save_model_name = "clip_vision_embedding"
        self.clip_vision_model_name_or_path = clip_vision_model_name_or_path
        self.projector_type = projector_type
        self.projector_depth = projector_depth
        self.projector_name_or_path = projector_name_or_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_additional_post_layernorm = use_additional_post_layernorm
        self.select_layer = select_layer
        self.embed_hidden_size = embed_hidden_size
        self.freeze_clip_vision_model = freeze_clip_vision_model
        self.freeze_embedding_layers = freeze_embedding_layers
        self.freeze_projector = freeze_projector

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            clip_vision_model_name_or_path, local_files_only=local_files_only
        )
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            clip_vision_model_name_or_path, local_files_only=local_files_only
        )

        projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.projector = build_projector(
            projector_cfg, in_hidden_size=self.clip_vision_model.config.hidden_size, out_hidden_size=embed_hidden_size, bias=True
        )

        self._init_weights(self.projector)

        self.post_layernorm = (
            nn.LayerNorm(embed_hidden_size, eps=self.clip_vision_model.config.layer_norm_eps)
            if use_additional_post_layernorm
            else nn.Identity()
        )

        self.image_embed_len = (self.clip_vision_model.config.image_size // self.clip_vision_model.config.patch_size) ** 2

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)

        if self.projector.load_model(projector_name_or_path):
            logger.info(f">>> loading `CLIPVisionEmbedding` projector from {projector_name_or_path}")

        self.clip_vision_model.requires_grad_(not freeze_clip_vision_model)
        if not freeze_clip_vision_model:
            for i in range(select_layer + 1, 0):
                self.clip_vision_model.vision_model.encoder.layers[i].requires_grad_(False)
            self.clip_vision_model.vision_model.post_layernorm.requires_grad_(False)

        # NOTE: must be set after `self.clip_vision_model.requires_grad_(not freeze_clip_vision_model)`
        self.clip_vision_model.vision_model.embeddings.requires_grad_(not freeze_embedding_layers)

        self.projector.requires_grad_(not freeze_projector)

    @property
    def processor(self):
        return self.clip_image_processor

    @property
    def embed_len(self):
        return self.image_embed_len

    @property
    def embed_dim(self):
        return self.embed_hidden_size

    @property
    def config(self) -> dict:
        return dict(
            clip_vision_model_name_or_path=self.clip_vision_model_name_or_path,
            clip_vision_model_config=self.clip_vision_model.config.to_dict(),
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            select_layer=self.select_layer,
            embed_len=self.embed_len,
            embed_dim=self.embed_dim,
            freeze_clip_vision_model=self.freeze_clip_vision_model,
            freeze_embedding_layers=self.freeze_embedding_layers,
            freeze_projector=self.freeze_projector,
        )

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_clip_vision_model:
            ignored_modules.append(self.clip_vision_model)
        if self.freeze_projector:
            ignored_modules.append(self.projector)
        return ignored_modules

    def save_model(self, output_dir: str):
        logger.info(f"saving `CLIPVisionEmbedding`...")
        torch.save(self.state_dict(), os.path.join(output_dir, f"{self.save_model_name}.bin"))

    def load_model(self, output_dir: str):
        if check_path_and_file(output_dir, f"{self.save_model_name}.bin"):
            logger.info(f">>> loading `CLIPVisionEmbedding`... from {output_dir}")
            loading_status = self.load_state_dict(torch.load(os.path.join(output_dir, f"{self.save_model_name}.bin"), map_location="cpu"))
            logger.info(f"{loading_status}")
        # HACK: For compatibility
        if check_path_and_file(output_dir, f"clip_vision_model_projector.pt"):
            self.projector.load_state_dict(
                torch.load(os.path.join(output_dir, "clip_vision_model_projector.pt"), map_location="cpu")
            )

    def forward(self, images: torch.FloatTensor | None = None):
        # HACK: dummy forward to avoid `find_unused_parameters` error
        is_dummy = images is None
        if is_dummy:
            c, h, w = 3, self.processor.crop_size["height"], self.processor.crop_size["width"]
            images = torch.zeros(1, c, h, w, device=self.clip_vision_model.device, dtype=self.clip_vision_model.dtype)

        output = self.clip_vision_model(images, output_hidden_states=True)
        hidden_state = output.hidden_states[self.select_layer]
        image_features = hidden_state[:, 1:]

        image_embeds = self.projector(image_features)[-1]
        image_embeds = self.post_layernorm(image_embeds)

        if is_dummy:
            return (0.0 * image_embeds).sum()
        else:
            return image_embeds
