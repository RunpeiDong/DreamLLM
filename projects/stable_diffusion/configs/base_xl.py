from omegaconf import OmegaConf

from omni.constants import MODEL_ZOOS
from omni.utils.import_utils import (
    is_basemind_platform_available,
    is_shaipower_platform_available,
    is_volc_mlplatform_available,
)

local_files_only = is_volc_mlplatform_available() or is_basemind_platform_available() or is_shaipower_platform_available()

config = OmegaConf.create(flags={"allow_objects": True})

config.model = dict(
    model_name_or_path=MODEL_ZOOS["stabilityai/stable-diffusion-xl-base-1.0"],
    vae_model_name_or_path=MODEL_ZOOS["madebyollin/sdxl-vae-fp16-fix"],
    local_files_only=local_files_only,
    enable_xformers_memory_efficient_attention=True,
)

config.data = dict(
    datasets=["blip_laion", "laion400m", "laion_coco"],
    size_list=["8M", "11M", "11M"],
    resolution=1024,
    center_crop=True,
    random_flip=True,
)

config.training = dict(
    output_dir="./work_dirs/sdxlbase10_ft",
    per_gpu_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=1e-6,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    num_train_epochs=1,
    lr_scheduler_type="constant",
    warmup_steps=0,
    save_steps=2000,
    save_total_limit=10,
    fp16=True,
    tf32=True,
    report_to="wandb",
    run_project="text2image-fine-tune-sdxl",
    run_name="sd_run_name",
    gradient_checkpointing=False,
    proportion_empty_prompts=0.2,
    input_perturbation=0.0,
    noise_offset=0.0,
    snr_gamma=None,
    use_ema=False,
    prediction_type=None,
    validation_steps=2000,
    validation_prompts="a cute Sundar Pichai creature",
    num_validation_images=4,
    scale_lr=False,
    timestep_bias_strategy="none",
    timestep_bias_multiplier=1.0,
    timestep_bias_begin=0,
    timestep_bias_end=1000,
    timestep_bias_portion=0.25,
)
