from contextlib import ExitStack
from typing import Iterator, Tuple

import numpy as np
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from pydantic import field_validator
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import DEFAULT_PRECISION, LATENT_SCALE_FACTOR
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation, get_scheduler
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.latents_to_image import LatentsToImageInvocation
from invokeai.app.invocations.model import UNetField, VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.invocations.tiled_multi_diffusion_denoise_latents import crop_controlnet_data
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.lora import LoRAModelRaw
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    PipelineIntermediateState,
    image_resized_to_grid_as_tensor,
)
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.tiles.tiles import (
    calc_tiles_min_overlap,
    merge_tiles_with_linear_blending,
)
from invokeai.backend.tiles.utils import TBLR, Tile
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "tiled_stable_diffusion_refine",
    title="Tiled Stable Diffusion Refine",
    tags=["upscale", "denoise"],
    category="latents",
    classification=Classification.Beta,
    version="1.0.0",
)
class TiledStableDiffusionRefineInvocation(BaseInvocation):
    """A tiled Stable Diffusion pipeline for refining high resolution images. This invocation is intended to be used to
    refine an image after upscaling i.e. it is the second step in a typical "tiled upscaling" workflow.

    The same result can be achieved by constructing a workflow, but that workflow would require 'iterate' nodes. The
    main reason that this invocation exists is so that this workflow can be run without 'iterate' nodes - which have
    some disadvantages and aren't permitted in the hosted InvokeAI app.
    """

    image: ImageField = InputField(description="Image to be refined.")

    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    noise: LatentsField = InputField(
        description=FieldDescriptions.noise,
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
        description="Target overlap between adjacent tiles in image space.",
    )
    steps: int = InputField(default=18, gt=0, description=FieldDescriptions.steps)
    cfg_scale: float | list[float] = InputField(default=6.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
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
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    vae_fp32: bool = InputField(
        default=DEFAULT_PRECISION == torch.float32, description="Whether to use float32 precision when running the VAE."
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

    def _scale_tile(self, tile: Tile, scale: int) -> Tile:
        """Scale the tile by the given factor."""
        return Tile(
            coords=TBLR(
                top=tile.coords.top * scale,
                bottom=tile.coords.bottom * scale,
                left=tile.coords.left * scale,
                right=tile.coords.right * scale,
            ),
            overlap=TBLR(
                top=tile.overlap.top * scale,
                bottom=tile.overlap.bottom * scale,
                left=tile.overlap.left * scale,
                right=tile.overlap.right * scale,
            ),
        )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Convert tile image-space dimensions to latent-space dimensions.
        latent_tile_height = self.tile_height // LATENT_SCALE_FACTOR
        latent_tile_width = self.tile_width // LATENT_SCALE_FACTOR
        latent_tile_overlap = self.tile_overlap // LATENT_SCALE_FACTOR

        # Load the input image.
        input_image = context.images.get_pil(self.image.image_name)
        # Convert the input image to a torch.Tensor.
        input_image_torch = image_resized_to_grid_as_tensor(input_image.convert("RGB"), multiple_of=LATENT_SCALE_FACTOR)
        input_image_torch = input_image_torch.unsqueeze(0)  # Add a batch dimension.
        # Validate our assumptions about the shape of input_image_torch.
        batch_size, channels, image_height, image_width = input_image_torch.shape
        assert batch_size == 1
        assert channels == 3

        # Load the noise tensor.
        noise = context.tensors.load(self.noise.latents_name)
        if list(noise.shape) != [
            batch_size,
            4,
            image_height // LATENT_SCALE_FACTOR,
            image_width // LATENT_SCALE_FACTOR,
        ]:
            raise ValueError(
                f"Incompatible noise and image dimensions. Image shape: {input_image_torch.shape}. "
                f"Noise shape: {noise.shape}. Expected noise shape: [1, 1, "
                f"{image_height // LATENT_SCALE_FACTOR}, {image_width // LATENT_SCALE_FACTOR}]. "
            )
        latent_height, latent_width = noise.shape[2:]

        # Extract the seed from the noise field.
        assert self.noise.seed is not None
        seed = self.noise.seed or 0

        # Calculate the tile locations in both latent space and image space.
        latent_space_tiles = calc_tiles_min_overlap(
            image_height=latent_height,
            image_width=latent_width,
            tile_height=latent_tile_height,
            tile_width=latent_tile_width,
            min_overlap=latent_tile_overlap,
        )
        image_space_tiles = [self._scale_tile(tile, LATENT_SCALE_FACTOR) for tile in latent_space_tiles]

        # Split the input image into tiles in torch.Tensor format.
        image_tiles_torch: list[torch.Tensor] = []
        for tile in image_space_tiles:
            image_tile = input_image_torch[
                :,
                :,
                tile.coords.top : tile.coords.bottom,
                tile.coords.left : tile.coords.right,
            ]
            image_tiles_torch.append(image_tile)

        # VAE-encode each image tile independently.
        vae_info = context.models.load(self.vae.vae)
        latent_tiles: list[torch.Tensor] = []
        for image_tile_torch in image_tiles_torch:
            latent_tiles.append(
                ImageToLatentsInvocation.vae_encode(
                    vae_info=vae_info, upcast=self.vae_fp32, tiled=False, image_tensor=image_tile_torch
                )
            )

        # Crop the global noise into tiles.
        noise_tiles: list[torch.Tensor] = []
        for tile in latent_space_tiles:
            noise_tile = noise[
                :,
                :,
                tile.coords.top : tile.coords.bottom,
                tile.coords.left : tile.coords.right,
            ]
            noise_tiles.append(noise_tile)

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

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
                latent_height=latent_tile_height,
                latent_width=latent_tile_width,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                cfg_rescale_multiplier=self.cfg_rescale_multiplier,
            )

            controlnet_data = DenoiseLatentsInvocation.prep_control_data(
                context=context,
                control_input=self.control,
                # NOTE: We use the shape of the global noise tensor here, because this is a global ControlNet. We tile
                # it later.
                latents_shape=list(noise.shape),
                # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                do_classifier_free_guidance=True,
                exit_stack=exit_stack,
            )

            # Split the controlnet_data into tiles.
            # controlnet_data_tiles[t][c] is the c'th control data for the t'th tile.
            controlnet_data_tiles: list[list[ControlNetData]] = []
            for tile in latent_space_tiles:
                tile_controlnet_data = [crop_controlnet_data(cn, tile.coords) for cn in controlnet_data or []]
                controlnet_data_tiles.append(tile_controlnet_data)

            # Denoise (i.e. "refine") each tile independently.
            for latent_tile, noise_tile, controlnet_data_tile in tqdm(
                zip(latent_tiles, noise_tiles, controlnet_data_tiles, strict=True),
                desc="Refining tiles",
                total=len(latent_tiles),
            ):
                assert latent_tile.shape == noise_tile.shape

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
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    conditioning_data=conditioning_data,
                    control_data=controlnet_data_tile,
                    ip_adapter_data=None,
                    t2i_adapter_data=None,
                    callback=step_callback,
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
                tile_size=0,
            )
            refined_image_tiles.append(refined_image_tile)

        # TODO(ryand): I copied this from DenoiseLatentsInvocation. I'm not sure if it's actually important.
        TorchDevice.empty_cache()

        # Merge the refined image tiles back into a single image.
        refined_image_tiles_np = [np.array(t) for t in refined_image_tiles]
        merged_image_np = np.zeros(shape=(input_image.height, input_image.width, 3), dtype=np.uint8)
        merge_tiles_with_linear_blending(
            dst_image=merged_image_np,
            tiles=image_space_tiles,
            tile_images=refined_image_tiles_np,
            blend_amount=self.tile_overlap,
        )

        # Save the refined image and return its reference.
        merged_image_pil = Image.fromarray(merged_image_np)
        image_dto = context.images.save(image=merged_image_pil)

        return ImageOutput.build(image_dto)
