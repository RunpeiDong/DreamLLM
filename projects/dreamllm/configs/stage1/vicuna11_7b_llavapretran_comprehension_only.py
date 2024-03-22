from ..common import (
    clip_vision_embedding_config_init_kwargs,
    dream_embedding_config_init_kwargs,
    stable_diffusion_head_config_init_kwargs,
)
from .base import config

# Vision Encoder
clip_vision_embedding_config_init_kwargs.freeze_clip_vision_model = True
clip_vision_embedding_config_init_kwargs.freeze_embedding_layers = True  # freeze all patch, class, and position embeddings
clip_vision_embedding_config_init_kwargs.freeze_projector = False
# Diffusion Decoder
stable_diffusion_head_config_init_kwargs.freeze_vae = True
stable_diffusion_head_config_init_kwargs.freeze_unet = True
stable_diffusion_head_config_init_kwargs.freeze_projector = True
dream_embedding_config_init_kwargs.freeze_dream_queries = True

# update configs
config.model.loss_weight_lm = 1.0
config.model.loss_weight_vm = 0.0
config.model.plugins_config_init_kwargs = dict(
    clip_vision_embedding=clip_vision_embedding_config_init_kwargs,
    dream_embedding=dream_embedding_config_init_kwargs,
    stable_diffusion_head=stable_diffusion_head_config_init_kwargs,
)

config.data = dict(
    datasets=["llava_pretrain", "laion400m_orig", "laion_coco", "blip_laion", "gqa"],
    size_list=["558K", "20M", "20M", "20M", 13532530],
    comprehension_only=True, # NOTE
    creation_only=False, # NOTE
)

config.training = dict(
    output_dir="./work_dirs/dreamllm_stage1_comprehension_only_output_dir",
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
