import json
import os

from omegaconf import OmegaConf

from ..common import (
    clip_vision_embedding_config_init_kwargs,
    local_files_only,
)

"""
torchrun --nproc-per-node=8 -m projects.dreamllm.train \
--config_file projects/dreamllm/configs/sft/base.py \
"training.per_device_train_batch_size=16" \
"training.output_dir='./your_dir'"
"""

config = OmegaConf.create(flags={"allow_objects": True})

model_name_or_path = "path2model"

# load Vision Encoder connector weights
clip_vision_embedding_config_init_kwargs.pretrained_model_name_or_path = model_name_or_path

with open(os.path.join(model_name_or_path, "config.json")) as f:
    model_config = json.load(f)
hidden_size = model_config["hidden_size"]
max_position_embeddings = model_config["max_position_embeddings"]

clip_vision_embedding_config_init_kwargs.embed_hidden_size = hidden_size

# Vision Encoder
clip_vision_embedding_config_init_kwargs.freeze_clip_vision_model = True
clip_vision_embedding_config_init_kwargs.freeze_embedding_layers = True  # freeze all patch, class, and position embeddings
clip_vision_embedding_config_init_kwargs.freeze_projector = False

config.model = dict(
    model_name_or_path=model_name_or_path,
    model_max_length=max_position_embeddings,
    local_files_only=local_files_only,
    special_tokens_dict={},
    average_init_embed_tokens=False,
    freeze_embed_tokens=False,
    freeze_lm_model=False,
    freeze_lm_head=False,
    loss_weight_lm=1.0,
    plugins_config_init_kwargs=dict(
        clip_vision_embedding=clip_vision_embedding_config_init_kwargs,
    ),
)

config.data = dict(
    datasets=["llavav1.5_instruct"],
    datasets_init_kwargs=dict(seed=42),
    size_list=[],
)

config.training = dict(
    output_dir="./work_dirs/sft",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    learning_rate=4e-5,
    num_train_epochs=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=3e-3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    tf32=True,
    dataloader_num_workers=8,
    remove_unused_columns=False,
    fsdp="shard_grad_op auto_wrap",
    fsdp_config=dict(
        fsdp_transformer_layer_cls_to_wrap=["DreamLLMDecoderLayer"],
    ),
    optim="adamw_torch",
    report_to=["wandb"],
    run_project="llava",
    run_name="sft",
    gradient_checkpointing=True,
)
