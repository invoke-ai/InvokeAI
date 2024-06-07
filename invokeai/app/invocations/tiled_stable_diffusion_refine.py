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
    LatentsField,
    UIType,
)
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.latents_to_image import LatentsToImageInvocation
from invokeai.app.invocations.model import UNetField, VAEField
from invokeai.app.invocations.noise import get_noise
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.tiles.utils import Tile
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "tiled_stable_diffusion_refine",
    title="Tiled Stable Diffusion Refine",
    tags=["upscale", "denoise"],
    category="latents",
    version="1.0.0",
)
class TiledStableDiffusionRefine(BaseInvocation):
    """A tiled Stable Diffusion pipeline for refining high resolution images. This invocation is intended to be used to
    refine an image after upscaling i.e. it is the second step in a typical "tiled upscaling" workflow.
    """

    # Implementation order:
    # - Basic tiled denoising. Support text prompts, but no other features.
    # - Support LoRA + TI
    # - Support ControlNet
    # - IP-Adapter? (It has to run on each tile independently. Could be complicated to support batching.)

    image: ImageField = InputField(description="Image to be refined.")

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
    steps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps)
    cfg_scale: float | list[float] = InputField(default=7.5, description=FieldDescriptions.cfg_scale, title="CFG Scale")
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
    # control: Optional[Union[ControlField, list[ControlField]]] = InputField(
    #     default=None,
    #     input=Input.Connection,
    # )
    cfg_rescale_multiplier: float = InputField(
        title="CFG Rescale Multiplier", default=0, ge=0, lt=1, description=FieldDescriptions.cfg_rescale_multiplier
    )
    latents: LatentsField | None = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    vae_fp32: bool = InputField(
        default=DEFAULT_PRECISION == torch.float32, description="Whether to use float32 precision when running the VAE."
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
        assert latents.dim == 4  # We expect: (batch_size, channels, height, width).

        top = image_tile.coords.top // LATENT_SCALE_FACTOR
        left = image_tile.coords.left // LATENT_SCALE_FACTOR
        bottom = image_tile.coords.bottom // LATENT_SCALE_FACTOR
        right = image_tile.coords.right // LATENT_SCALE_FACTOR
        return latents[..., top:bottom, left:right]

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # TODO(ryand): Expose the seed parameter.
        seed = 0

        # Load the input image.
        input_image = context.images.get_pil(self.image.image_name)
        input_image_torch = image_resized_to_grid_as_tensor(input_image.convert("RGB"), multiple_of=LATENT_SCALE_FACTOR)

        # Calculate the tile locations to cover the image.
        # TODO(ryand): Expose these tiling parameters. (Keep in mind the multiple-of constraints on these params.)
        tiles = calc_tiles_min_overlap(
            image_height=input_image.height,
            image_width=input_image.width,
            tile_height=512,
            tile_width=512,
            min_overlap=128,
        )

        # Validate our assumptions about the shape of input_image_torch.
        assert input_image_torch.dim() == 4  # We expect: (batch_size, channels, height, width).
        assert input_image_torch.shape[:2] == (1, 3)

        # Split the input image into tiles.
        image_tiles: list[torch.Tensor] = []
        for tile in tiles:
            image_tile = input_image_torch[
                :,
                :,
                tile.coords.top : tile.coords.bottom,
                tile.coords.left : tile.coords.right,
            ]
            image_tiles.append(image_tile)

        # VAE-encode each image tile independently.
        # TODO(ryand): Is there any advantage to VAE-encoding the entire image before splitting it into tiles? What
        # about for decoding?
        vae_info = context.models.load(self.vae.vae)
        latent_tiles: list[torch.Tensor] = []
        for image_tile in image_tiles:
            latent_tiles.append(
                ImageToLatentsInvocation.vae_encode(
                    vae_info=vae_info, upcast=self.vae_fp32, tiled=False, image_tensor=image_tile
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

        # Load the UNet model.
        unet_info = context.models.load(self.unet.unet)

        refined_latent_tiles: list[torch.Tensor] = []
        with unet_info as unet:
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

            # Denoise (i.e. "refine") each tile independently.
            for latent_tile, noise_tile in zip(latent_tiles, noise_tiles, strict=True):
                assert latent_tile.shape == noise_tile.shape

                num_inference_steps, timesteps, init_timestep, scheduler_step_kwargs = (
                    DenoiseLatentsInvocation.init_scheduler(
                        scheduler,
                        device=unet.device,
                        steps=self.steps,
                        denoising_start=self.denoising_start,
                        denoising_end=self.denoising_end,
                        seed=seed,
                    )
                )

                refined_latent_tile = pipeline.latents_from_embeddings(
                    latents=latent_tile,
                    timesteps=timesteps,
                    init_timestep=init_timestep,
                    noise=noise_tile,
                    seed=seed,
                    mask=None,
                    masked_latents=None,
                    gradient_mask=None,
                    num_inference_steps=num_inference_steps,
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    conditioning_data=conditioning_data,
                    control_data=None,
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

        # Merge the refined image tiles back into a single image.
        ...

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
