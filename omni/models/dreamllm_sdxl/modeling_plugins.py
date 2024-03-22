import random
from typing import Any, Callable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import crop

from omni.models.dreamllm.modeling_plugins import StableDiffusionHead
from omni.models.projector.builder import build_projector
from omni.utils.loguru import logger


class SDXLDataProcessor:
    """Image processor for SDXL."""

    def __init__(self, resolution=1024, center_crop=False, random_flip=False):
        # for image processor
        self.resolution = resolution
        # TODO: ablate if center crop and random flip are necessary or beneficial
        self.center_crop = center_crop  # SDXL most likely uses random cropping, but we try to use center crop here
        self.random_flip = random_flip  # SDXL may used random flip, but we try not to do it now
        self.train_resize = T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR)
        self.train_crop = T.CenterCrop(resolution) if self.center_crop else T.RandomCrop(resolution)
        self.train_flip = T.RandomHorizontalFlip(p=1.0)
        self.image_processor_ldm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __call__(self, image):
        # image for dm
        original_size = [image.height, image.width]
        image = self.train_resize(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - self.resolution) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, (self.resolution, self.resolution))
            image = crop(image, y1, x1, h, w)
        if self.random_flip and random.random() < 0.5:
            # flip
            x1 = image.width - x1
            image = self.train_flip(image)
        crop_top_left = [y1, x1]
        return self.image_processor_ldm(image), list(original_size + crop_top_left + [self.resolution, self.resolution])


class StableDiffusionXLHead(StableDiffusionHead):
    def __init__(
        self,
        diffusion_name_or_path: str,
        pretrained_model_name_or_path: str = None,
        projector_type="linear",
        projector_depth: int = 1,
        projector_name_or_path: str = None,
        embed_hidden_size: int = 4096,
        global_condition_hidden_size: int = 1280,
        drop_prob: float | None = None,
        noise_offset: float = 0.0,
        input_perturbation: float = 0.0,
        snr_gamma: float | None = None,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        freeze_vae: bool = True,
        freeze_unet: bool = True,
        freeze_projector: bool = False,
        local_files_only: bool = True,
    ):
        super().__init__(
            diffusion_name_or_path=diffusion_name_or_path,
            pretrained_model_name_or_path=None,
            projector_type=projector_type,
            projector_depth=projector_depth,
            projector_name_or_path=projector_name_or_path,
            embed_hidden_size=embed_hidden_size,
            drop_prob=drop_prob,
            noise_offset=noise_offset,
            input_perturbation=input_perturbation,
            snr_gamma=snr_gamma,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
            freeze_vae=freeze_vae,
            freeze_unet=freeze_unet,
            freeze_projector=freeze_projector,
            local_files_only=local_files_only,
        )
        self.save_model_name = "stable_diffusion_xl_head"
        self.global_condition_hidden_size = global_condition_hidden_size

        # add a new global projectors, similar to two CLIP encoders as used in unCLIP & SDXL
        projector_cfg = dict(
            projector=projector_type,
            freeze_projector=freeze_projector,
            depth=projector_depth,
            save_model_name=self.save_model_name,
            model_name_or_path=None,
        )
        self.global_projector = build_projector(projector_cfg, in_hidden_size=embed_hidden_size, out_hidden_size=self.global_condition_hidden_size)
        if not self.global_projector.load_model(projector_name_or_path):
            self._init_weights(self.global_projector)

        if pretrained_model_name_or_path is not None:
            self.load_model(pretrained_model_name_or_path)

        self.global_projector.requires_grad_(not freeze_projector)

    @property
    def processor(self):
        return SDXLDataProcessor(resolution=1024, center_crop=False, random_flip=False)

    @property
    def config(self):
        return dict(
            diffusion_name_or_path=self.diffusion_name_or_path,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            embed_hidden_size=self.embed_hidden_size,
            global_condition_hidden_size=self.global_condition_hidden_size,
            drop_prob=self.drop_prob,
            noise_offset=self.noise_offset,
            input_perturbation=self.input_perturbation,
            snr_gamma=self.snr_gamma,
            freeze_vae=self.freeze_vae,
            freeze_unet=self.freeze_unet,
            freeze_projector=self.freeze_projector,
        )

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        if self.freeze_vae:
            ignored_modules.append(self.vae)
        if self.freeze_unet:
            ignored_modules.append(self.unet)
        if self.freeze_projector:
            ignored_modules.append(self.projector)
            ignored_modules.append(self.global_projector)

        return ignored_modules

    def to(self, device=None, dtype=None):
        # modified for avoiding avoid changing dtype of vae
        if dtype is not None and device is not None:
            self.unet.to(device=device, dtype=dtype)
            self.projector.to(device=device, dtype=dtype)
            self.global_projector.to(device=device, dtype=dtype)

        logger.warning("VAE will be kept dtype float32 and is not changed to dtype: {}.".format(dtype))

        return self

    def forward(
        self,
        images: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        u_encoder_hidden_states: torch.FloatTensor | None = None,
        add_time_ids: torch.FloatTensor | None = None,
        dream_embeddings: torch.FloatTensor | None = None,
    ):
        # HACK: avoid `find_unused_parameters` error
        is_dummy = images == None
        if is_dummy:
            assert dream_embeddings is not None, "You must provide `dream_embeddings` when dummy forward."
            dummy_image_features = torch.zeros(1, dream_embeddings.shape[1], self.embed_hidden_size, device=self.device, dtype=self.dtype)
            dummy_image_features = self.projector(dummy_image_features)[-1]
            dummy_global_image_features = self.global_projector(dummy_image_features)[-1]
            return (0.0 * dummy_image_features).sum() + (0.0 * dummy_global_image_features).sum() + (0.0 * dream_embeddings).sum()

        # Convert images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        assert (
            encoder_hidden_states.shape[0] == latents.shape[0]
        ), f"encoder_hidden_states.shape[0]: {encoder_hidden_states.shape[0]} != latents.shape[0]: {latents.shape[0]}"
        bsz = latents.shape[0]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn((bsz, latents.shape[1], 1, 1), device=latents.device)
        if self.input_perturbation:
            new_noise = noise + self.input_perturbation * torch.randn_like(noise)

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        global_encoder_hidden_states = encoder_hidden_states.mean(1)
        # encoder_hidden_states = encoder_hidden_states[:, :-1, ...]
        global_encoder_hidden_states = self.global_projector(global_encoder_hidden_states)[-1]
        encoder_hidden_states = self.projector(encoder_hidden_states)[-1]

        # TODO: support training with classifier free guidance (https://arxiv.org/abs/2207.12598)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        unet_added_conditions = {"time_ids": add_time_ids}
        unet_added_conditions.update({"text_embeds": global_encoder_hidden_states.float()})

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.float(), added_cond_kwargs=unet_added_conditions).sample

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self._compute_snr(timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        if is_dummy:
            loss = 0.0 * loss

        return loss

    @torch.no_grad()
    def pipeline(
        self,
        # prompt: str | list[str] | MultimodalContent = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        # negative_prompt: str | list[str] | MultimodalContent | None = None,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: Literal["latent", "pt", "np", "pil"] | None = "pil",
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        callback_steps: int = 1,
        cross_attention_kwargs: dict[str, Any] | None = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = [0, 0],
        target_size: Optional[Tuple[int, int]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # NOTE: The decoder only considers embedding inputs, so there is no raw prompt.
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(height, width, callback_steps, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        batch_size = prompt_embeds.shape[0]

        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # NOTE: After mounting to LLM, LLM takes on the task.
        assert prompt_embeds is not None, "`prompt_embeds` must be provided by LLM."
        global_prompt_embeds = prompt_embeds.mean(1)
        # prompt_embeds = prompt_embeds[:, :-1, ...]

        global_prompt_embeds = self.global_projector(global_prompt_embeds)[-1]
        prompt_embeds = self.projector(prompt_embeds)[-1]

        add_text_embeds = global_prompt_embeds
        if original_size is None:
            original_size = [self.unet.config.sample_size * (2 ** (len(self.vae.config.block_out_channels) - 1)),
                             self.unet.config.sample_size * (2 ** (len(self.vae.config.block_out_channels) - 1))]
        if target_size is None:
            target_size = original_size
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).to(prompt_embeds.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            # negative_prompt_embeds = self.projector(negative_prompt_embeds)[-1]
            assert negative_prompt_embeds is not None, "When using classifier free guidance, `negative_prompt_embeds` must be provided by LLM."
            global_negative_prompt_embeds = negative_prompt_embeds.mean(1)
            # negative_prompt_embeds = negative_prompt_embeds[:, :-1, ...]
            global_negative_prompt_embeds = self.global_projector(global_negative_prompt_embeds)[-1]
            negative_prompt_embeds = self.projector(negative_prompt_embeds)[-1]

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([global_negative_prompt_embeds, global_prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        # 4. Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self._rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        self.vae.to(dtype=torch.float32)
        latents = latents.float()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        return image
