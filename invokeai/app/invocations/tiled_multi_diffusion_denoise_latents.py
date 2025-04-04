import copy
from contextlib import ExitStack
from typing import Iterator, Tuple

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.controlnet import ControlField
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation, get_scheduler
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.model import UNetField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusers_pipeline import ControlNetData, PipelineIntermediateState
from invokeai.backend.stable_diffusion.multi_diffusion_pipeline import (
    MultiDiffusionPipeline,
    MultiDiffusionRegionConditioning,
)
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.tiles.tiles import (
    calc_tiles_min_overlap,
)
from invokeai.backend.tiles.utils import TBLR
from invokeai.backend.util.devices import TorchDevice


def crop_controlnet_data(control_data: ControlNetData, latent_region: TBLR) -> ControlNetData:
    """Crop a ControlNetData object to a region."""
    # Create a shallow copy of the control_data object.
    control_data_copy = copy.copy(control_data)
    # The ControlNet reference image is the only attribute that needs to be cropped.
    control_data_copy.image_tensor = control_data.image_tensor[
        :,
        :,
        latent_region.top * LATENT_SCALE_FACTOR : latent_region.bottom * LATENT_SCALE_FACTOR,
        latent_region.left * LATENT_SCALE_FACTOR : latent_region.right * LATENT_SCALE_FACTOR,
    ]
    return control_data_copy


@invocation(
    "tiled_multi_diffusion_denoise_latents",
    title="Tiled Multi-Diffusion Denoise - SD1.5, SDXL",
    tags=["upscale", "denoise"],
    category="latents",
    version="1.0.1",
)
class TiledMultiDiffusionDenoiseLatents(BaseInvocation):
    """Tiled Multi-Diffusion denoising.

    This node handles automatically tiling the input image, and is primarily intended for global refinement of images
    in tiled upscaling workflows. Future Multi-Diffusion nodes should allow the user to specify custom regions with
    different parameters for each region to harness the full power of Multi-Diffusion.

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
    tile_height: int = InputField(
        default=1024, gt=0, multiple_of=LATENT_SCALE_FACTOR, description="Height of the tiles in image space."
    )
    tile_width: int = InputField(
        default=1024, gt=0, multiple_of=LATENT_SCALE_FACTOR, description="Width of the tiles in image space."
    )
    tile_overlap: int = InputField(
        default=32,
        multiple_of=LATENT_SCALE_FACTOR,
        gt=0,
        description="The overlap between adjacent tiles in pixel space. (Of course, tile merging is applied in latent "
        "space.) Tiles will be cropped during merging (if necessary) to ensure that they overlap by exactly this "
        "amount.",
    )
    steps: int = InputField(default=18, gt=0, description=FieldDescriptions.steps)
    cfg_scale: float | list[float] = InputField(default=6.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    denoising_start: float = InputField(
        default=0.0,
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

    @staticmethod
    def create_pipeline(
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
    ) -> MultiDiffusionPipeline:
        # TODO(ryand): Get rid of this FakeVae hack.
        class FakeVae:
            class FakeVaeConfig:
                def __init__(self) -> None:
                    self.block_out_channels = [0]

            def __init__(self) -> None:
                self.config = FakeVae.FakeVaeConfig()

        return MultiDiffusionPipeline(
            vae=FakeVae(),
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Convert tile image-space dimensions to latent-space dimensions.
        latent_tile_height = self.tile_height // LATENT_SCALE_FACTOR
        latent_tile_width = self.tile_width // LATENT_SCALE_FACTOR
        latent_tile_overlap = self.tile_overlap // LATENT_SCALE_FACTOR

        seed, noise, latents = DenoiseLatentsInvocation.prepare_noise_and_latents(context, self.noise, self.latents)
        _, _, latent_height, latent_width = latents.shape

        # Calculate the tile locations to cover the latent-space image.
        # TODO(ryand): In the future, we may want to revisit the tile overlap strategy. Things to consider:
        # - How much overlap 'context' to provide for each denoising step.
        # - How much overlap to use during merging/blending.
        # - Should we 'jitter' the tile locations in each step so that the seams are in different places?
        tiles = calc_tiles_min_overlap(
            image_height=latent_height,
            image_width=latent_width,
            tile_height=latent_tile_height,
            tile_width=latent_tile_width,
            min_overlap=latent_tile_overlap,
        )

        # Get the unet's config so that we can pass the base to sd_step_callback().
        unet_config = context.models.get_config(self.unet.unet.key)

        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        # Prepare an iterator that yields the UNet's LoRA models and their weights.
        def _lora_loader() -> Iterator[Tuple[ModelPatchRaw, float]]:
            for lora in self.unet.loras:
                lora_info = context.models.load(lora.lora)
                assert isinstance(lora_info.model, ModelPatchRaw)
                yield (lora_info.model, lora.weight)
                del lora_info

        device = TorchDevice.choose_torch_device()
        with (
            ExitStack() as exit_stack,
            context.models.load(self.unet.unet) as unet,
            LayerPatcher.apply_smart_model_patches(
                model=unet, patches=_lora_loader(), prefix="lora_unet_", dtype=unet.dtype
            ),
        ):
            assert isinstance(unet, UNet2DConditionModel)
            latents = latents.to(device=device, dtype=unet.dtype)
            if noise is not None:
                noise = noise.to(device=device, dtype=unet.dtype)
            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.unet.scheduler,
                scheduler_name=self.scheduler,
                seed=seed,
                unet_config=unet_config,
            )
            pipeline = self.create_pipeline(unet=unet, scheduler=scheduler)

            # Prepare the prompt conditioning data. The same prompt conditioning is applied to all tiles.
            conditioning_data = DenoiseLatentsInvocation.get_conditioning_data(
                context=context,
                positive_conditioning_field=self.positive_conditioning,
                negative_conditioning_field=self.negative_conditioning,
                device=device,
                dtype=unet.dtype,
                latent_height=latent_tile_height,
                latent_width=latent_tile_width,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                cfg_rescale_multiplier=self.cfg_rescale_multiplier,
            )

            controlnet_data = DenoiseLatentsInvocation.prep_control_data(
                context=context,
                control_input=self.control,
                latents_shape=list(latents.shape),
                device=device,
                # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                do_classifier_free_guidance=True,
                exit_stack=exit_stack,
            )

            # Split the controlnet_data into tiles.
            # controlnet_data_tiles[t][c] is the c'th control data for the t'th tile.
            controlnet_data_tiles: list[list[ControlNetData]] = []
            for tile in tiles:
                tile_controlnet_data = [crop_controlnet_data(cn, tile.coords) for cn in controlnet_data or []]
                controlnet_data_tiles.append(tile_controlnet_data)

            # Prepare the MultiDiffusionRegionConditioning list.
            multi_diffusion_conditioning: list[MultiDiffusionRegionConditioning] = []
            for tile, tile_controlnet_data in zip(tiles, controlnet_data_tiles, strict=True):
                multi_diffusion_conditioning.append(
                    MultiDiffusionRegionConditioning(
                        region=tile,
                        text_conditioning_data=conditioning_data,
                        control_data=tile_controlnet_data,
                    )
                )

            timesteps, init_timestep, scheduler_step_kwargs = DenoiseLatentsInvocation.init_scheduler(
                scheduler,
                device=device,
                steps=self.steps,
                denoising_start=self.denoising_start,
                denoising_end=self.denoising_end,
                seed=seed,
            )

            # Run Multi-Diffusion denoising.
            result_latents = pipeline.multi_diffusion_denoise(
                multi_diffusion_conditioning=multi_diffusion_conditioning,
                target_overlap=latent_tile_overlap,
                latents=latents,
                scheduler_step_kwargs=scheduler_step_kwargs,
                noise=noise,
                timesteps=timesteps,
                init_timestep=init_timestep,
                callback=step_callback,
            )

        result_latents = result_latents.to("cpu")
        # TODO(ryand): I copied this from DenoiseLatentsInvocation. I'm not sure if it's actually important.
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
