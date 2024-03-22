import json
import os

from omegaconf import OmegaConf

from omni.constants import MODEL_ZOOS
from omni.models.dreamllm.tokenization_dreamllm import special_tokens_dict

from ..common import (
    clip_vision_embedding_config_init_kwargs,
    dream_embedding_config_init_kwargs,
    local_files_only,
    stable_diffusion_head_config_init_kwargs,
)

config = OmegaConf.create(flags={"allow_objects": True})

model_name_or_path = MODEL_ZOOS["lmsys/vicuna-7b-v1.1"]
with open(os.path.join(model_name_or_path, "config.json")) as f:
    model_config = json.load(f)
hidden_size = model_config["hidden_size"]
max_position_embeddings = model_config["max_position_embeddings"]

dream_embedding_config_init_kwargs.embed_hidden_size = hidden_size
clip_vision_embedding_config_init_kwargs.embed_hidden_size = hidden_size
stable_diffusion_head_config_init_kwargs.embed_hidden_size = hidden_size

# Vision Encoder
clip_vision_embedding_config_init_kwargs.freeze_clip_vision_model = True
clip_vision_embedding_config_init_kwargs.freeze_embedding_layers = True  # freeze all patch, class, and position embeddings
clip_vision_embedding_config_init_kwargs.freeze_projector = True
# Diffusion Decoder
stable_diffusion_head_config_init_kwargs.freeze_vae = True
stable_diffusion_head_config_init_kwargs.freeze_unet = True
stable_diffusion_head_config_init_kwargs.freeze_projector = False
dream_embedding_config_init_kwargs.freeze_dream_queries = False

# 2-layer MLP Linear-GLEU-Linear
# clip_vision_embedding_config_init_kwargs.projector_type = "mlp"
# clip_vision_embedding_config_init_kwargs.projector_depth = 2

config.model = dict(
    model_name_or_path=model_name_or_path,
    model_max_length=max_position_embeddings,
    local_files_only=local_files_only,
    special_tokens_dict=special_tokens_dict,
    average_init_embed_tokens=False,
    freeze_embed_tokens=True,
    freeze_lm_model=True,
    freeze_lm_head=True,
    loss_weight_lm=0.0,
    loss_weight_vm=1.0,
    plugins_config_init_kwargs=dict(
        clip_vision_embedding=clip_vision_embedding_config_init_kwargs,
        dream_embedding=dream_embedding_config_init_kwargs,
        stable_diffusion_head=stable_diffusion_head_config_init_kwargs,
    ),
)

config.data = dict(
    datasets=["laion400m_orig", "laion_coco", "blip_laion"],
    size_list=["20M", "20M", "20M"],
    datasets_init_kwargs=dict(seed=42),
    comprehension_only=False, # NOTE
    creation_only=True, # NOTE
)

config.training = dict(
    output_dir="./work_dirs/dreamllm_stage1_creation_only_output_dir",
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
    run_project="dreamllm",
    run_name="wandb_run_name",
    gradient_checkpointing=True,
)
