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
    model_name_or_path=MODEL_ZOOS["stabilityai/stable-diffusion-2-1-base"],
    local_files_only=local_files_only,
    enable_xformers_memory_efficient_attention=True,
)

config.data = dict(
    datasets=["blip_laion", "laion400m", "laion_coco"],
    size_list=["8M", "11M", "11M"],
    resolution=512,
    center_crop=True,
    random_flip=True,
)

config.training = dict(
    output_dir="./work_dirs/sd21base_bliplaion_laion400m_laioncoco",
    per_gpu_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
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
    run_project="train_text_to_image",
    run_name="sd_run_name",
    gradient_checkpointing=False,
    input_perturbation=0.0,
    noise_offset=0.0,
    snr_gamma=None,
    use_ema=True,
    prediction_type=None,
    validation_steps=2000,
    validation_prompts=[
        "A couple of glasses are sitting on a table.",
        "A teddybear on a skateboard in Times Square.",
        "A green sign that says 'Very Deep Learning' and is at the edge of the Grand Canyon. Puffy white clouds are in the sky."
        "an armchair in the shape of an avocado",
    ],
    scale_lr=False,
)
