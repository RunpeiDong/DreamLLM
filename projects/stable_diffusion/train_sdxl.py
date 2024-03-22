#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""

import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection, PretrainedConfig

from omni.config.arg_parser import LazyAguments, LazyArgumentParser
from omni.data.constants import DataManager
from omni.data.manager.dataset_type import DatasetType
from omni.train.training_args import TrainingArguments as _TrainingArguments
from omni.utils.import_utils import is_accelerate_available, is_wandb_available, is_xformers_available
from omni.utils.loguru import logger
from omni.utils.training_utils import EMAModel, get_optimizer, get_scheduler

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.23.0")


def save_model_card(config, repo_id: str, images=None, validation_prompts=None, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {config.model.model_name_or_path}
dataset: {config.data.datasets}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
inference: true
---
    """

    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{config.model.model_name_or_path}** on the **{config.data.datasets}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompts}: \n
{img_str}

Special VAE used for training: {config.model.vae_model_name_or_path}.

## Training info

These are the key hyperparameters used during training:

* Epochs: {config.training.num_train_epochs}
* Learning rate: {config.training.learning_rate}
* Batch size: {config.training.per_gpu_train_batch_size}
* Gradient accumulation steps: {config.training.gradient_accumulation_steps}
* Image resolution: {config.data.resolution}
* Mixed-precision: {os.environ.get("ACCELERATE_MIXED_PRECISION", "no")}

"""

    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


# fmt: off
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
# fmt: on


def log_validation(vae_path, unet, config, accelerator, weight_dtype, step):
    logger.info(f"Running validation in {step} step...")
    logger.info(f"Generating {config.training.num_validation_images} images with prompt: {config.training.validation_prompts}")

    # create pipeline
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if config.model.vae_model_name_or_path is None else None,
        revision=config.model.revision,
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.model.model_name_or_path,
        vae=vae,
        unet=accelerator.unwrap_model(unet),
        revision=config.model.revision,
        torch_dtype=weight_dtype,
    )
    if config.training.prediction_type is not None:
        scheduler_args = {"prediction_type": config.training.prediction_type}
        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(config.training.seed) if config.training.seed else None

    validation_list = []
    for idx in range(len(config.training.validation_prompts)):
        with torch.cuda.amp.autocast():
            images = [
                pipeline(prompt=config.training.validation_prompts[idx], generator=generator, num_inference_steps=25).images[0]
                for _ in range(config.training.num_validation_images)
            ]
        validation_list.extend(
            [wandb.Image(image, caption=f"{i}: {config.training.validation_prompts[idx]}") for i, image in enumerate(images)]
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            logger.warning("image logging not implemented for tensorboard")
            continue
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log({"validation": validation_list}, step=step)
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def compute_vae_encodings(images, vae):
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts):
    prompt_embeds_list = []
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# fmt: off
def generate_timestep_weights(config, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(config.training.timestep_bias_portion * num_timesteps)

    if config.training.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif config.training.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif config.training.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = config.training.timestep_bias_begin
        range_end = config.training.timestep_bias_end
        if range_begin < 0:
            raise ValueError("When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero.")
        if range_end > num_timesteps:
            raise ValueError("When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps.")
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if config.training.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= config.training.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights
# fmt: on


# fmt: off
@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    vae_model_name_or_path: str | None = field(default=None, metadata={"help": "Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038."})
    revision: str | None = field(default=None, metadata={"help": "Revision of pretrained model identifier from huggingface.co/models."})
    local_files_only: bool = field(default=False, metadata={"help": "Whether to only use local files."})
    enable_xformers_memory_efficient_attention: bool = field(default=False, metadata={"help": "Whether or not to use xformers."})


@dataclass
class DataArguments:
    datasets: list[str] = field(default_factory=list, metadata={"help": "The datasets to use for training."})
    size_list: list[str | int] = field(default_factory=list, metadata={"help": "The size of each dataset."})
    resolution: int = field(default=1024, metadata={"help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."})
    center_crop: bool = field(default=False, metadata={"help": "Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping."})
    random_flip: bool = field(default=False, metadata={"help": "Whether to randomly flip images horizontally."})


@dataclass
class TrainingArguments(_TrainingArguments):
    proportion_empty_prompts: float = field(default=0.0, metadata={"help": "Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."})
    input_perturbation: float = field(default=0.0, metadata={"help": "The scale of input perturbation. Recommended 0.1."})
    noise_offset: float = field(default=0.0, metadata={"help": "The scale of noise offset."})
    snr_gamma: float | None = field(default=None, metadata={"help": "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."})
    prediction_type: str | None = field(
        default=None,
        metadata={
            "help": "The prediction_type that shall be used for training."
            "Choose between 'epsilon' or 'v_prediction' or leave `None`."
            "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen."
        },
    )
    validation_steps: int = field(default=500, metadata={"help": "Run validation every X steps."})
    validation_prompts: str | list[str] | None = field(default=None, metadata={"help": "A prompt that is used during validation to verify that the model is learning."})
    num_validation_images: int = field(default=4, metadata={"help": "Number of images that should be generated during validation with `validation_prompts`."})
    timestep_bias_strategy: str = field(
        default="none",
        metadata={
            "help": "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        },
    )
    timestep_bias_multiplier: float = field(
        default=1.0,
        metadata={
            "help": "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        },
    )
    timestep_bias_begin: int = field(
        default=0,
        metadata={
            "help": "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        },
    )
    timestep_bias_end: int = field(
        default=1000,
        metadata={
            "help": "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        },
    )
    timestep_bias_portion: float = field(
        default=0.25,
        metadata={
            "help": "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        },
    )


@dataclass
class Arguments(LazyAguments):
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)
# fmt: on


@logger.catch
def main():
    config = LazyArgumentParser(Arguments)

    mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
    if config.training.fp16:
        mixed_precision_dtype = "fp16"
    elif config.training.bf16:
        mixed_precision_dtype = "bf16"
    os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=mixed_precision_dtype,
        log_with=config.training.report_to,
        project_config=ProjectConfiguration(project_dir=config.training.output_dir),
    )
    logger.info(str(accelerator.state))

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.training.push_to_hub:
            repo_id = create_repo(
                repo_id=config.training.hub_model_id or Path(config.training.output_dir).name,
                exist_ok=True,
                token=config.training.hub_token,
            ).repo_id

    # fmt: off
    # Load scheduler, tokenizer, text_encoder and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.model.model_name_or_path, subfolder="scheduler")

    tokenizer_one = AutoTokenizer.from_pretrained(config.model.model_name_or_path, subfolder="tokenizer", revision=config.model.revision, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(config.model.model_name_or_path, subfolder="tokenizer_2", revision=config.model.revision, use_fast=False)
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(config.model.model_name_or_path, config.model.revision, subfolder="text_encoder")
    text_encoder_cls_two = import_model_class_from_model_name_or_path(config.model.model_name_or_path, config.model.revision, subfolder="text_encoder_2")
    text_encoder_one = text_encoder_cls_one.from_pretrained(config.model.model_name_or_path, subfolder="text_encoder", revision=config.model.revision)
    text_encoder_two = text_encoder_cls_two.from_pretrained(config.model.model_name_or_path, subfolder="text_encoder_2", revision=config.model.revision)

    vae_path = config.model.model_name_or_path if config.model.vae_model_name_or_path is None else config.model.vae_model_name_or_path
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae" if config.model.vae_model_name_or_path is None else None, revision=config.model.revision)
    unet = UNet2DConditionModel.from_pretrained(config.model.model_name_or_path, subfolder="unet", revision=config.model.revision)

    # Freeze vae and text encoders and set unet as trainable
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.train()

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    # Create EMA for the unet.
    if config.training.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(config.model.model_name_or_path, subfolder="unet", revision=config.model.revision)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
    # fmt: on

    if config.model.enable_xformers_memory_efficient_attention:
        if is_xformers_available(">=", "0.0.17"):
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if is_accelerate_available(">=", "0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.training.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.training.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.training.scale_lr:
        config.training.learning_rate = (
            config.training.learning_rate
            * config.training.gradient_accumulation_steps
            * config.training.per_gpu_train_batch_size
            * accelerator.num_processes
        )

    # Optimizer creation
    optimizer = get_optimizer(
        unet.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.weight_decay,
        eps=config.training.adam_epsilon,
    )

    # Get the datasets
    train_dataset = DataManager(datasets=config.data.datasets, datasets_init_kwargs={}, size_list=config.data.size_list)

    def collate_fn(examples):
        raw_images = []
        captions = []
        for example in examples:
            assert example.dataset_type == DatasetType.ImageTextPair, "Dataset (sample) type must be `ImageTextPair`"
            raw_images.append(example.image)
            captions.append(example.text)
        return {"raw_images": raw_images, "captions": captions}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.training.per_gpu_train_batch_size,
        num_workers=config.training.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    override_max_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    if config.training.max_steps < 0:
        config.training.max_steps = config.training.num_train_epochs * num_update_steps_per_epoch
        override_max_steps = True

    lr_scheduler = get_scheduler(
        config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps * accelerator.num_processes,
        num_training_steps=config.training.max_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    if override_max_steps:
        config.training.max_steps = config.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.training.num_train_epochs = math.ceil(config.training.max_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = asdict(config)
        accelerator.init_trackers(
            config.training.run_project,
            tracker_config,
            {"wandb": {"name": config.training.run_name}},
        )

    # Train!
    total_batch_size = (
        config.training.per_gpu_train_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.training.per_gpu_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.training.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.training.resume_from_checkpoint:
        if config.training.resume_from_checkpoint != "latest":
            path = os.path.basename(config.training.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.training.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.training.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.training.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.training.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.training.max_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # fmt: off
    # Preprocessing the datasets.
    train_resize = transforms.Resize(config.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(config.data.resolution) if config.data.center_crop else transforms.RandomCrop(config.data.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # fmt: on

    def process_batch_raw_images(batch: dict):
        raw_images: list[PIL.Image.Image] = batch.pop("raw_images")
        images = [image.convert("RGB") for image in raw_images]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if config.data.center_crop:
                y1 = max(0, int(round((image.height - config.data.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - config.data.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (config.data.resolution, config.data.resolution))
                image = crop(image, y1, x1, h, w)
            if config.data.random_flip and random.random() < 0.5:
                # flip
                x1 = image.width - x1
                image = train_flip(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        batch["original_sizes"] = original_sizes
        batch["crop_top_lefts"] = crop_top_lefts
        batch["pixel_values"] = all_images

        return batch

    if isinstance(config.training.validation_prompts, str):
        config.training.validation_prompts = [config.training.validation_prompts]

    # log original model
    if accelerator.is_main_process:
        if config.training.validation_prompts is not None and global_step == 0:
            if config.training.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            log_validation(vae_path, unet, config, accelerator, weight_dtype, global_step)

    for epoch in range(first_epoch, config.training.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # fmt: off
            # train step
            with accelerator.accumulate(unet):
                batch = process_batch_raw_images(batch)

                _images = batch.pop("pixel_values")
                model_input = compute_vae_encodings(_images, vae)
                model_input = model_input.to(accelerator.device)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                if config.training.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.training.noise_offset * torch.randn((model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device)

                bsz = model_input.shape[0]
                if config.training.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(config, noise_scheduler.config.num_train_timesteps).to(model_input.device)
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (config.data.resolution, config.data.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat([compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])])
                unet_added_conditions = {"time_ids": add_time_ids}

                _prompt_batch = batch.pop("captions")
                prompt_embeds, pooled_prompt_embeds = encode_prompt(_prompt_batch, text_encoders, tokenizers, config.training.proportion_empty_prompts)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                # Predict the noise residual
                model_pred = unet(noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample

                # Get the target for loss depending on the prediction type
                if config.training.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.training.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if config.training.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = torch.stack([snr, config.training.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.training.per_gpu_train_batch_size)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # fmt: on

            # fmt: off
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save state
                    if global_step % config.training.save_steps == 0:
                        # _before_ saving state, check if this save would set us over the `save_total_limit`
                        if config.training.save_total_limit is not None:
                            checkpoints = os.listdir(config.training.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.training.save_total_limit:
                                num_to_remove = len(checkpoints) - config.training.save_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.training.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # log validation
                    if config.training.validation_prompts is not None and (global_step - 1) % config.training.validation_steps == 0:
                        if config.training.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        log_validation(vae_path, unet, config, accelerator, weight_dtype, global_step)
            # fmt: on

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.training.max_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if config.training.use_ema:
            ema_unet.copy_to(unet.parameters())

        # Serialize pipeline.
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if config.model.vae_model_name_or_path is None else None,
            revision=config.model.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            config.model.model_name_or_path,
            unet=unet,
            vae=vae,
            revision=config.model.revision,
            torch_dtype=weight_dtype,
        )
        if config.training.prediction_type is not None:
            scheduler_args = {"prediction_type": config.training.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(config.training.output_dir)

        # fmt: off
        # Run a final round of inference.
        all_images = []
        if config.training.validation_prompts and config.training.num_validation_images > 0:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(config.training.seed) if config.training.seed else None

            test_list = []
            for idx in range(len(config.training.validation_prompts)):
                with torch.cuda.amp.autocast():
                    images = [
                        pipeline(config.training.validation_prompts[idx], num_inference_steps=25, generator=generator).images[0]
                        for _ in range(config.training.num_validation_images)
                    ]
                    all_images.extend(images)
                test_list.extend([wandb.Image(image, caption=f"{i}: {config.training.validation_prompts[idx]}") for i, image in enumerate(images)])

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    logger.warning("image logging not implemented for tensorboard")
                    continue
                    # np_images = np.stack([np.asarray(img) for img in images])
                    # tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                elif tracker.name == "wandb":
                    tracker.log({"test": test_list})
        # fmt: on

        if config.training.push_to_hub:
            save_model_card(
                config,
                repo_id=repo_id,
                images=all_images,
                validation_prompts=config.training.validation_prompts,
                repo_folder=config.training.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=config.training.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
