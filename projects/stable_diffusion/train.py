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
"""Fine-tuning script for Stable Diffusion for text2image."""

import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, make_image_grid
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

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


def save_model_card(config, repo_id: str, images=None, repo_folder=None):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(config.training.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {config.model.model_name_or_path}
datasets:
- {config.data.datasets}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{config.model.model_name_or_path}** on the **{config.data.datasets}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {config.training.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{config.training.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

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


def log_validation(vae, text_encoder, tokenizer, unet, config, accelerator, weight_dtype, epoch):
    logger.info("Running validation...")

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model.model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=config.model.revision,
        torch_dtype=weight_dtype,
        local_files_only=config.model.local_files_only,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if config.model.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=accelerator.device).manual_seed(config.training.seed) if config.training.seed else None

    images = []
    for i in range(len(config.training.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(config.training.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {config.training.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    revision: str | None = field(
        default=None, metadata={"help": "Revision of pretrained model identifier from huggingface.co/models."}
    )
    local_files_only: bool = field(default=False, metadata={"help": "Whether to only use local files."})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False, metadata={"help": "Whether or not to use xformers."}
    )


@dataclass
class DataArguments:
    datasets: list[str] = field(default_factory=list, metadata={"help": "The datasets to use for training."})
    size_list: list[str | int] = field(default_factory=list, metadata={"help": "The size of each dataset."})
    resolution: int = field(
        default=512,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
    )
    center_crop: bool = field(
        default=False,
        metadata={
            "help": "Whether to center crop the input images to the resolution. "
            "If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping."
        },
    )
    random_flip: bool = field(default=False, metadata={"help": "Whether to randomly flip images horizontally."})


@dataclass
class TrainingArguments(_TrainingArguments):
    input_perturbation: float = field(default=0.0, metadata={"help": "The scale of input perturbation. Recommended 0.1."})
    noise_offset: float = field(default=0.0, metadata={"help": "The scale of noise offset."})
    snr_gamma: float | None = field(
        default=None,
        metadata={
            "help": "SNR weighting gamma to be used if rebalancing the loss."
            "Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."
        },
    )
    prediction_type: str | None = field(
        default=None,
        metadata={
            "help": "The prediction_type that shall be used for training."
            "Choose between 'epsilon' or 'v_prediction' or leave `None`."
            "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen."
        },
    )
    validation_steps: int = field(default=500, metadata={"help": "Run validation every X steps."})
    validation_prompts: list[str] | None = field(
        default=None, metadata={"help": "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."}
    )


@dataclass
class Arguments(LazyAguments):
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)


"""
torchrun --master_addr $MLP_WORKER_0_HOST --master_port $MLP_WORKER_0_PORT --node_rank $MLP_ROLE_INDEX --nnodes $MLP_WORKER_NUM --nproc_per_node $MLP_WORKER_GPU \
-m projects.stable_diffusion.train \
--config_file ./projects/stable_diffusion/configs/base.py \

"""


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
    tokenizer = CLIPTokenizer.from_pretrained(config.model.model_name_or_path, subfolder="tokenizer", revision=config.model.revision)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(config.model.model_name_or_path, subfolder="text_encoder", revision=config.model.revision)
        vae = AutoencoderKL.from_pretrained(config.model.model_name_or_path, subfolder="vae", revision=config.model.revision)

    unet = UNet2DConditionModel.from_pretrained(config.model.model_name_or_path, subfolder="unet", revision=config.model.revision)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

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

    optimizer = get_optimizer(
        unet.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.weight_decay,
        eps=config.training.adam_epsilon,
    )

    # Get the datasets
    train_dataset = DataManager(datasets=config.data.datasets, datasets_init_kwargs={}, size_list=config.data.size_list)

    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.data.resolution)
            if config.data.center_crop
            else transforms.RandomCrop(config.data.resolution),
            transforms.RandomHorizontalFlip() if config.data.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        pixel_values = []
        captions = []
        for example in examples:
            assert example.dataset_type == DatasetType.ImageTextPair, "Dataset (sample) type must be `ImageTextPair`"
            pixel_values.append(train_transforms(example.image))
            captions.append(example.text)
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

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

    if config.training.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

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

    for epoch in range(first_epoch, config.training.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.training.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.training.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if config.training.input_perturbation:
                    new_noise = noise + config.training.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.training.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if config.training.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.training.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

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
                    mse_loss_weights = (
                        torch.stack([snr, config.training.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.training.per_gpu_train_batch_size)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.training.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % config.training.save_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `save_total_limit`
                        if config.training.save_total_limit is not None:
                            checkpoints = os.listdir(config.training.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `save_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.training.save_total_limit:
                                num_to_remove = len(checkpoints) - config.training.save_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.training.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.training.max_steps:
                break

            if accelerator.is_main_process:
                if (
                    config.training.validation_prompts is not None
                    and num_update_steps_per_epoch * epoch + step % config.training.validation_steps == 0
                ):
                    if config.training.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    log_validation(vae, text_encoder, tokenizer, unet, config, accelerator, weight_dtype, global_step)
                    if config.training.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if config.training.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model.model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=config.model.revision,
        )
        pipeline.save_pretrained(config.training.output_dir)

        # Run a final round of inference.
        images = []
        if config.training.validation_prompts is not None:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if config.model.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if config.training.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(config.training.seed)

            # fmt: off
            for i in range(len(config.training.validation_prompts)):
                with torch.autocast("cuda"):
                    image = pipeline(config.training.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)
            # fmt: on

        if config.training.push_to_hub:
            save_model_card(config, repo_id, images, repo_folder=config.training.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=config.training.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
