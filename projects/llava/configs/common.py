from omni.constants import MODEL_ZOOS
from omni.models.llava.configuration_llava import ConfigAndInitKwargs, create_config_init_kwargs
from omni.models.llava.modeling_plugins import CLIPVisionEmbedding
from omni.utils.import_utils import is_volc_mlplatform_available

if is_volc_mlplatform_available():
    local_files_only = True
else:
    local_files_only = False

# NOTE: don't forget add `embed_hidden_size` kwargs
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
