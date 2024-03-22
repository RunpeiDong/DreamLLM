import json
import os

from omegaconf import OmegaConf

from omni.constants import MODEL_ZOOS
from omni.models.llava.tokenization_llava import special_tokens_dict

from ..common import (
    clip_vision_embedding_config_init_kwargs,
    local_files_only,
)

config = OmegaConf.create(flags={"allow_objects": True})

model_name_or_path = MODEL_ZOOS["lmsys/vicuna-7b-v1.1"]
with open(os.path.join(model_name_or_path, "config.json")) as f:
    model_config = json.load(f)
hidden_size = model_config["hidden_size"]
max_position_embeddings = model_config["max_position_embeddings"]

clip_vision_embedding_config_init_kwargs.embed_hidden_size = hidden_size

# Vision Encoder
clip_vision_embedding_config_init_kwargs.freeze_clip_vision_model = True
clip_vision_embedding_config_init_kwargs.freeze_embedding_layers = True  # freeze all patch, class, and position embeddings
clip_vision_embedding_config_init_kwargs.freeze_projector = False
# Diffusion Decoder

config.model = dict(
    model_name_or_path=model_name_or_path,
    model_max_length=max_position_embeddings,
    local_files_only=local_files_only,
    special_tokens_dict=special_tokens_dict,
    average_init_embed_tokens=False,
    freeze_embed_tokens=True,
    freeze_lm_model=True,
    freeze_lm_head=True,
    loss_weight_lm=1.0,
    plugins_config_init_kwargs=dict(
        clip_vision_embedding=clip_vision_embedding_config_init_kwargs,
    ),
)

# config.data = dict(
#     datasets=["llava_pretrain", "laion400m_orig"],
#     size_list=["558K", "20M"],
#     comprehension_only=True,
#     creation_only=False,
# )

config.data = dict(
    datasets=["llava_pretrain"],
    size_list=["558K"],
)

config.training = dict(
    output_dir="./work_dirs/llava_align/",
    vit_llrd=False,
    llm_llrd=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    learning_rate=2e-3,
    num_train_epochs=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=3e-3,
    logging_steps=10,
    save_steps=2000,
    save_total_limit=3,
    bf16=True,
    tf32=True,
    dataloader_num_workers=8,
    remove_unused_columns=False,
    optim="adamw_torch",
    report_to=["wandb"],
    run_project="llava",
    run_name="pretrain",
    gradient_checkpointing=True,
)
