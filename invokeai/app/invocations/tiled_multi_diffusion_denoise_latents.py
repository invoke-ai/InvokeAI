from contextlib import ExitStack
from typing import Iterator, Tuple

import numpy as np
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR, SCHEDULER_NAME_VALUES
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation, get_scheduler
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.latents_to_image import LatentsToImageInvocation
from invokeai.app.invocations.model import UNetField
from invokeai.app.invocations.noise import get_noise
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.lora import LoRAModelRaw
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.stable_diffusion.diffusers_pipeline import ControlNetData
from invokeai.backend.tiles.tiles import (
    calc_tiles_min_overlap,
    merge_tiles_with_linear_blending,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "tiled_stable_diffusion_refine",
    title="Tiled Stable Diffusion Refine",
    tags=["upscale", "denoise"],
    category="latents",
    # TODO(ryand): Reset to 1.0.0 right before release.
    version="1.0.0",
)
class TiledMultiDiffusionDenoiseLatents(BaseInvocation):
    """Tiled Multi-Diffusion denoising.

    This node handles automatically tiling the input image. Future iterations of
    this node should allow the user to specify custom regions with different parameters for each region to harness the
    full power of Multi-Diffusion.

    This node has a similar interface to the `DenoiseLatents` node, but it has a reduced feature set (no IP-Adapter,
    T2I-Adapter, masking, etc.).
    """

    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    noise: LatentsField | None = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    latents: LatentsField | None = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    # TODO(ryand): Add multiple-of validation.
    # TODO(ryand): Smaller defaults might make more sense.
    tile_height: int = InputField(default=112, gt=0, description="Height of the tiles in latent space.")
    tile_width: int = InputField(default=112, gt=0, description="Width of the tiles in latent space.")
    tile_min_overlap: int = InputField(
        default=16,
        gt=0,
        description="The minimum overlap between adjacent tiles in latent space. The actual overlap may be larger than "
        "this to evenly cover the entire image.",
    )
    steps: int = InputField(default=18, gt=0, description=FieldDescriptions.steps)
    cfg_scale: float | list[float] = InputField(default=6.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    # TODO(ryand): The default here should probably be 0.0.
    denoising_start: float = InputField(
        default=0.65,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    scheduler: SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )
    unet: UNetField = InputField(
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    cfg_rescale_multiplier: float = InputField(
        title="CFG Rescale Multiplier", default=0, ge=0, lt=1, description=FieldDescriptions.cfg_rescale_multiplier
    )
    control: ControlField | list[ControlField] | None = InputField(
        default=None,
        input=Input.Connection,
    )

    @field_validator("cfg_scale")
    def ge_one(cls, v: list[float] | float) -> list[float] | float:
        """Validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError("cfg_scale must be greater than 1")
        else:
            if v < 1:
                raise ValueError("cfg_scale must be greater than 1")
        return v

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        seed, noise, latents = DenoiseLatentsInvocation.prepare_noise_and_latents(context, self.noise, self.latents)
        _, _, latent_height, latent_width = latents.shape

        # If noise is None, populate it here.
        # TODO(ryand): Currently there is logic to generate noise deeper in the stack if it is None. We should just move
        # that logic up the stack in all places that it's relied upon (i.e. do it in prepare_noise_and_latents). In this
        # particular case, we want to make sure that the noise is generated globally rather than per-tile so that
        # overlapping tile regions use the same noise.
        if noise is None:
            noise = get_noise(
                width=latent_width * LATENT_SCALE_FACTOR,
                height=latent_height * LATENT_SCALE_FACTOR,
                device=TorchDevice.choose_torch_device(),
                seed=seed,
                downsampling_factor=LATENT_SCALE_FACTOR,
                use_cpu=True,
            )

        # Calculate the tile locations to cover the latent-space image.
        # TODO(ryand): Add constraints on the tile params. Is there a multiple-of constraint?
        tiles = calc_tiles_min_overlap(
            image_height=latent_height,
            image_width=latent_width,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            min_overlap=self.tile_min_overlap,
        )

        # Split the noise and latents into tiles.
        noise_tiles: list[torch.Tensor] = []
        latent_tiles: list[torch.Tensor] = []
        for tile in tiles:
            noise_tile = noise[..., tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right]
            latent_tile = latents[..., tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right]
            noise_tiles.append(noise_tile)
            latent_tiles.append(latent_tile)

        # Prepare an iterator that yields the UNet's LoRA models and their weights.
        def _lora_loader() -> Iterator[Tuple[LoRAModelRaw, float]]:
            for lora in self.unet.loras:
                lora_info = context.models.load(lora.lora)
                assert isinstance(lora_info.model, LoRAModelRaw)
                yield (lora_info.model, lora.weight)
                del lora_info

        # Load the UNet model.
        unet_info = context.models.load(self.unet.unet)

        refined_latent_tiles: list[torch.Tensor] = []
        with ExitStack() as exit_stack, unet_info as unet, ModelPatcher.apply_lora_unet(unet, _lora_loader()):
            assert isinstance(unet, UNet2DConditionModel)
            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.unet.scheduler,
                scheduler_name=self.scheduler,
                seed=seed,
            )
            pipeline = DenoiseLatentsInvocation.create_pipeline(unet=unet, scheduler=scheduler)

            # Prepare the prompt conditioning data. The same prompt conditioning is applied to all tiles.
            conditioning_data = DenoiseLatentsInvocation.get_conditioning_data(
                context=context,
                positive_conditioning_field=self.positive_conditioning,
                negative_conditioning_field=self.negative_conditioning,
                unet=unet,
                latent_height=self.tile_height,
                latent_width=self.tile_width,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                cfg_rescale_multiplier=self.cfg_rescale_multiplier,
            )

            controlnet_data = DenoiseLatentsInvocation.prep_control_data(
                context=context,
                control_input=self.control,
                latents_shape=list(latents.shape),
                # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                do_classifier_free_guidance=True,
                exit_stack=exit_stack,
            )

            # Split the controlnet_data into tiles.
            if controlnet_data is not None:
                # controlnet_data_tiles[t][c] is the c'th control data for the t'th tile.
                controlnet_data_tiles: list[list[ControlNetData]] = []
                for tile in tiles:
                    # To split the controlnet_data into tiles, we simply need to crop each image_tensor. All other
                    # params can be copied unmodified.
                    tile_controlnet_data = [
                        ControlNetData(
                            model=cn.model,
                            image_tensor=cn.image_tensor[
                                :,
                                :,
                                tile.coords.top * LATENT_SCALE_FACTOR : tile.coords.bottom * LATENT_SCALE_FACTOR,
                                tile.coords.left * LATENT_SCALE_FACTOR : tile.coords.right * LATENT_SCALE_FACTOR,
                            ],
                            weight=cn.weight,
                            begin_step_percent=cn.begin_step_percent,
                            end_step_percent=cn.end_step_percent,
                            control_mode=cn.control_mode,
                            resize_mode=cn.resize_mode,
                        )
                        for cn in controlnet_data
                    ]
                    controlnet_data_tiles.append(tile_controlnet_data)

            # Denoise (i.e. "refine") each tile independently.
            for image_tile_np, latent_tile, noise_tile in zip(image_tiles_np, latent_tiles, noise_tiles, strict=True):
                assert latent_tile.shape == noise_tile.shape

                # Prepare a PIL Image for ControlNet processing.
                # TODO(ryand): This is a bit awkward that we have to prepare both torch.Tensor and PIL.Image versions of
                # the tiles. Ideally, the ControlNet code should be able to work with Tensors.
                image_tile_pil = Image.fromarray(image_tile_np)

                timesteps, init_timestep, scheduler_step_kwargs = DenoiseLatentsInvocation.init_scheduler(
                    scheduler,
                    device=unet.device,
                    steps=self.steps,
                    denoising_start=self.denoising_start,
                    denoising_end=self.denoising_end,
                    seed=seed,
                )

                # TODO(ryand): Think about when/if latents/noise should be moved off of the device to save VRAM.
                latent_tile = latent_tile.to(device=unet.device, dtype=unet.dtype)
                noise_tile = noise_tile.to(device=unet.device, dtype=unet.dtype)
                refined_latent_tile = pipeline.latents_from_embeddings(
                    latents=latent_tile,
                    timesteps=timesteps,
                    init_timestep=init_timestep,
                    noise=noise_tile,
                    seed=seed,
                    mask=None,
                    masked_latents=None,
                    gradient_mask=None,
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    conditioning_data=conditioning_data,
                    control_data=[controlnet_data],
                    ip_adapter_data=None,
                    t2i_adapter_data=None,
                    callback=lambda x: None,
                )
                refined_latent_tiles.append(refined_latent_tile)

        # VAE-decode each refined latent tile independently.
        refined_image_tiles: list[Image.Image] = []
        for refined_latent_tile in refined_latent_tiles:
            refined_image_tile = LatentsToImageInvocation.vae_decode(
                context=context,
                vae_info=vae_info,
                seamless_axes=self.vae.seamless_axes,
                latents=refined_latent_tile,
                use_fp32=self.vae_fp32,
                use_tiling=False,
            )
            refined_image_tiles.append(refined_image_tile)

        # TODO(ryand): I copied this from DenoiseLatentsInvocation. I'm not sure if it's actually important.
        TorchDevice.empty_cache()

        # Merge the refined image tiles back into a single image.
        refined_image_tiles_np = [np.array(t) for t in refined_image_tiles]
        merged_image_np = np.zeros(shape=(input_image.height, input_image.width, 3), dtype=np.uint8)
        # TODO(ryand): Tune the blend_amount. Should this be exposed as a parameter?
        merge_tiles_with_linear_blending(
            dst_image=merged_image_np, tiles=tiles, tile_images=refined_image_tiles_np, blend_amount=self.tile_overlap
        )

        # Save the refined image and return its reference.
        merged_image_pil = Image.fromarray(merged_image_np)
        image_dto = context.images.save(image=merged_image_pil)

        return ImageOutput.build(image_dto)
