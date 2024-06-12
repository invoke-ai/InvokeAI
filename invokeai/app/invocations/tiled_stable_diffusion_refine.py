from contextlib import ExitStack
from typing import Iterator, Tuple

import numpy as np
import numpy.typing as npt
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import DEFAULT_PRECISION, LATENT_SCALE_FACTOR, SCHEDULER_NAME_VALUES
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation, get_scheduler
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    UIType,
)
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.latents_to_image import LatentsToImageInvocation
from invokeai.app.invocations.model import ModelIdentifierField, UNetField, VAEField
from invokeai.app.invocations.noise import get_noise
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.lora import LoRAModelRaw
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.stable_diffusion.diffusers_pipeline import ControlNetData, image_resized_to_grid_as_tensor
from invokeai.backend.tiles.tiles import calc_tiles_with_overlap, merge_tiles_with_linear_blending
from invokeai.backend.tiles.utils import Tile
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.hotfixes import ControlNetModel


@invocation(
    "tiled_stable_diffusion_refine",
    title="Tiled Stable Diffusion Refine",
    tags=["upscale", "denoise"],
    category="latents",
    version="1.0.0",
)
class TiledStableDiffusionRefineInvocation(BaseInvocation):
    """A tiled Stable Diffusion pipeline for refining high resolution images. This invocation is intended to be used to
    refine an image after upscaling i.e. it is the second step in a typical "tiled upscaling" workflow.
    """

    image: ImageField = InputField(description="Image to be refined.")

    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # TODO(ryand): Add multiple-of validation.
    tile_height: int = InputField(default=512, gt=0, description="Height of the tiles.")
    tile_width: int = InputField(default=512, gt=0, description="Width of the tiles.")
    tile_overlap: int = InputField(
        default=16,
        gt=0,
        description="Target overlap between adjacent tiles (the last row/column may overlap more than this).",
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
    # HACK(ryand): We probably want to allow the user to control all of the parameters in ControlField. But, we akwardly
    # don't want to use the image field. Figure out how best to handle this.
    # TODO(ryand): Currently, there is no ControlNet preprocessor applied to the tile images. In other words, we pretty
    # much assume that it is a tile ControlNet. We need to decide how we want to handle this. E.g. find a way to support
    # CN preprocessors, raise a clear warning when a non-tile CN model is selected, hardcode the supported CN models,
    # etc.
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model, ui_type=UIType.ControlNetModel
    )
    control_weight: float = InputField(default=0.6)

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
    def crop_latents_to_tile(latents: torch.Tensor, image_tile: Tile) -> torch.Tensor:
        """Crop the latent-space tensor to the area corresponding to the image-space tile.
        The tile coordinates must be divisible by the LATENT_SCALE_FACTOR.
        """
        for coord in [image_tile.coords.top, image_tile.coords.left, image_tile.coords.right, image_tile.coords.bottom]:
            if coord % LATENT_SCALE_FACTOR != 0:
                raise ValueError(
                    f"The tile coordinates must all be divisible by the latent scale factor"
                    f" ({LATENT_SCALE_FACTOR}). {image_tile.coords=}."
                )
        assert latents.dim() == 4  # We expect: (batch_size, channels, height, width).

        top = image_tile.coords.top // LATENT_SCALE_FACTOR
        left = image_tile.coords.left // LATENT_SCALE_FACTOR
        bottom = image_tile.coords.bottom // LATENT_SCALE_FACTOR
        right = image_tile.coords.right // LATENT_SCALE_FACTOR
        return latents[..., top:bottom, left:right]

    def run_controlnet(
        self,
        image: Image.Image,
        controlnet_model: ControlNetModel,
        weight: float,
        do_classifier_free_guidance: bool,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        control_mode: CONTROLNET_MODE_VALUES = "balanced",
        resize_mode: CONTROLNET_RESIZE_VALUES = "just_resize_simple",
    ) -> ControlNetData:
        control_image = prepare_control_image(
            image=image,
            do_classifier_free_guidance=do_classifier_free_guidance,
            width=width,
            height=height,
            device=device,
            dtype=dtype,
            control_mode=control_mode,
            resize_mode=resize_mode,
        )
        return ControlNetData(
            model=controlnet_model,
            image_tensor=control_image,
            weight=weight,
            begin_step_percent=0.0,
            end_step_percent=1.0,
            control_mode=control_mode,
            # Any resizing needed should currently be happening in prepare_control_image(), but adding resize_mode to
            # ControlNetData in case needed in the future.
            resize_mode=resize_mode,
        )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # TODO(ryand): Expose the seed parameter.
        seed = 0

        # Load the input image.
        input_image = context.images.get_pil(self.image.image_name)

        # Calculate the tile locations to cover the image.
        # We have selected this tiling strategy to make it easy to achieve tile coords that are multiples of 8. This
        # facilitates conversions between image space and latent space.
        # TODO(ryand): Expose these tiling parameters. (Keep in mind the multiple-of constraints on these params.)
        tiles = calc_tiles_with_overlap(
            image_height=input_image.height,
            image_width=input_image.width,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            overlap=self.tile_overlap,
        )

        # Convert the input image to a torch.Tensor.
        input_image_torch = image_resized_to_grid_as_tensor(input_image.convert("RGB"), multiple_of=LATENT_SCALE_FACTOR)
        input_image_torch = input_image_torch.unsqueeze(0)  # Add a batch dimension.
        # Validate our assumptions about the shape of input_image_torch.
        assert input_image_torch.dim() == 4  # We expect: (batch_size, channels, height, width).
        assert input_image_torch.shape[:2] == (1, 3)

        # Split the input image into tiles in torch.Tensor format.
        image_tiles_torch: list[torch.Tensor] = []
        for tile in tiles:
            image_tile = input_image_torch[
                :,
                :,
                tile.coords.top : tile.coords.bottom,
                tile.coords.left : tile.coords.right,
            ]
            image_tiles_torch.append(image_tile)

        # Split the input image into tiles in numpy format.
        # TODO(ryand): We currently maintain both np.ndarray and torch.Tensor tiles. Ideally, all operations should work
        # with torch.Tensor tiles.
        input_image_np = np.array(input_image)
        image_tiles_np: list[npt.NDArray[np.uint8]] = []
        for tile in tiles:
            image_tile_np = input_image_np[
                tile.coords.top : tile.coords.bottom,
                tile.coords.left : tile.coords.right,
                :,
            ]
            image_tiles_np.append(image_tile_np)

        # VAE-encode each image tile independently.
        # TODO(ryand): Is there any advantage to VAE-encoding the entire image before splitting it into tiles? What
        # about for decoding?
        vae_info = context.models.load(self.vae.vae)
        latent_tiles: list[torch.Tensor] = []
        for image_tile_torch in image_tiles_torch:
            latent_tiles.append(
                ImageToLatentsInvocation.vae_encode(
                    vae_info=vae_info, upcast=self.vae_fp32, tiled=False, image_tensor=image_tile_torch
                )
            )

        # Generate noise with dimensions corresponding to the full image in latent space.
        # It is important that the noise tensor is generated at the full image dimension and then tiled, rather than
        # generating for each tile independently. This ensures that overlapping regions between tiles use the same
        # noise.
        assert input_image_torch.shape[2] % LATENT_SCALE_FACTOR == 0
        assert input_image_torch.shape[3] % LATENT_SCALE_FACTOR == 0
        global_noise = get_noise(
            width=input_image_torch.shape[3],
            height=input_image_torch.shape[2],
            device=TorchDevice.choose_torch_device(),
            seed=seed,
            downsampling_factor=LATENT_SCALE_FACTOR,
            use_cpu=True,
        )

        # Crop the global noise into tiles.
        noise_tiles = [self.crop_latents_to_tile(latents=global_noise, image_tile=t) for t in tiles]

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
            # Assume that all tiles have the same shape.
            _, _, latent_height, latent_width = latent_tiles[0].shape
            conditioning_data = DenoiseLatentsInvocation.get_conditioning_data(
                context=context,
                positive_conditioning_field=self.positive_conditioning,
                negative_conditioning_field=self.negative_conditioning,
                unet=unet,
                latent_height=latent_height,
                latent_width=latent_width,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                cfg_rescale_multiplier=self.cfg_rescale_multiplier,
            )

            # Load the ControlNet model.
            # TODO(ryand): Support multiple ControlNet models.
            controlnet_model = exit_stack.enter_context(context.models.load(self.control_model))
            assert isinstance(controlnet_model, ControlNetModel)

            # Denoise (i.e. "refine") each tile independently.
            for image_tile_np, latent_tile, noise_tile in zip(image_tiles_np, latent_tiles, noise_tiles, strict=True):
                assert latent_tile.shape == noise_tile.shape

                # Prepare a PIL Image for ControlNet processing.
                # TODO(ryand): This is a bit awkward that we have to prepare both torch.Tensor and PIL.Image versions of
                # the tiles. Ideally, the ControlNet code should be able to work with Tensors.
                image_tile_pil = Image.fromarray(image_tile_np)

                # Run the ControlNet on the image tile.
                height, width, _ = image_tile_np.shape
                # The height and width must be evenly divisible by LATENT_SCALE_FACTOR. This is enforced earlier, but we
                # validate this assumption here.
                assert height % LATENT_SCALE_FACTOR == 0
                assert width % LATENT_SCALE_FACTOR == 0
                controlnet_data = self.run_controlnet(
                    image=image_tile_pil,
                    controlnet_model=controlnet_model,
                    weight=self.control_weight,
                    do_classifier_free_guidance=True,
                    width=width,
                    height=height,
                    device=controlnet_model.device,
                    dtype=controlnet_model.dtype,
                    control_mode="balanced",
                    resize_mode="just_resize_simple",
                )

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
