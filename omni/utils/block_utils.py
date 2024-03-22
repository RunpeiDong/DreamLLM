from __future__ import annotations

from omni.constants import *
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from typing import Literal


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


SchedulerType = Literal["ddpm", "ddim", "dpm"]


def build_diffusion_module(
    diffusion_model_name_or_path: str = MODEL_ZOOS["runwayml/stable-diffusion-v1-5"], scheduler_type: SchedulerType = "ddpm", local_files_only: bool = USE_HF_LOCAL_FILES,
    enable_xformers_memory_efficient_attention: bool = ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION,
) -> tuple[DDPMScheduler | DDIMScheduler | DPMSolverMultistepScheduler, AutoencoderKL, UNet2DConditionModel, CLIPTextModel, CLIPTokenizer]:
    # load SD scheduler, tokenizer and models
    if scheduler_type == "ddpm":
        noise_scheduler = DDPMScheduler.from_pretrained(diffusion_model_name_or_path, subfolder="scheduler", local_files_only=local_files_only)
    elif scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(diffusion_model_name_or_path, subfolder="scheduler", local_files_only=local_files_only)
    elif scheduler_type == "dpm":
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(diffusion_model_name_or_path, subfolder="scheduler", local_files_only=local_files_only)
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    vae = AutoencoderKL.from_pretrained(diffusion_model_name_or_path, subfolder="vae", revision=None, local_files_only=local_files_only)
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_name_or_path, subfolder="unet", revision=None, local_files_only=local_files_only)
    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_name_or_path, subfolder="text_encoder", revision=None)
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_name_or_path, subfolder="tokenizer", revision=None)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            import xformers
            from packaging import version

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    return noise_scheduler, vae, unet, text_encoder, tokenizer


def build_vision_module(
    clip_model_name_or_path: str = MODEL_ZOOS["openai/clip-vit-large-patch14"], return_vision_tower: bool = True, local_files_only: bool = USE_HF_LOCAL_FILES
) -> tuple[CLIPImageProcessor, CLIPVisionModel | None]:
    clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model_name_or_path, local_files_only=local_files_only)
    if return_vision_tower:
        clip_vision_tower = CLIPVisionModel.from_pretrained(clip_model_name_or_path, local_files_only=local_files_only)
        return clip_image_processor, clip_vision_tower
    return clip_image_processor, None
