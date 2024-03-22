from omni.constants import MODEL_ZOOS
from omni.models.dreamllm.configuration_dreamllm import ConfigAndInitKwargs, create_config_init_kwargs
from omni.models.dreamllm.modeling_plugins import CLIPVisionEmbedding, DreamEmbedding, StableDiffusionHead
from omni.utils.import_utils import is_volc_mlplatform_available

if is_volc_mlplatform_available():
    local_files_only = True
else:
    local_files_only = False

# NOTE: don't forget add `embed_hidden_size` kwargs
dream_embedding_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=DreamEmbedding,
        _name_="dream_embedding",
        _plugin_type_="embedding",
        pretrained_model_name_or_path=None,
        num_dream_queries=64,
        # embed_hidden_size=hidden_size,
        freeze_dream_queries=False,
    )
)
clip_vision_embedding_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=CLIPVisionEmbedding,
        _name_="clip_vision_embedding",
        _plugin_type_="embedding",
        projector_type="linear",
        projector_depth=1,
        clip_vision_model_name_or_path=MODEL_ZOOS["openai/clip-vit-large-patch14"],
        pretrained_model_name_or_path=None,
        # embed_hidden_size=hidden_size,
        use_additional_post_layernorm=False,
        select_layer=-2,
        freeze_clip_vision_model=True,
        freeze_embedding_layers=True,
        freeze_projector=False,
        local_files_only=local_files_only,
    )
)
stable_diffusion_head_config_init_kwargs = create_config_init_kwargs(
    ConfigAndInitKwargs(
        _class_=StableDiffusionHead,
        _name_="stable_diffusion_head",
        _plugin_type_="head",
        projector_type="linear",
        projector_depth=1,
        diffusion_name_or_path=MODEL_ZOOS["stabilityai/stable-diffusion-2-1-base"],
        pretrained_model_name_or_path=None,
        # embed_hidden_size=hidden_size,
        freeze_vae=True,
        freeze_unet=True,
        freeze_projector=False,
        local_files_only=local_files_only,
    )
)
