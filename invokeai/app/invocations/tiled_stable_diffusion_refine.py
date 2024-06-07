import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import DEFAULT_PRECISION, LATENT_SCALE_FACTOR, SCHEDULER_NAME_VALUES
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
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation, get_scheduler
from invokeai.app.invocations.model import UNetField, VAEField
from invokeai.app.invocations.noise import get_noise
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
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
        # TODO(ryand): Is there any advantage to VAE-encoding the entire image before splitting it into tiles?
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
        noise_tiles = get_noise(
            width=input_image_torch.shape[3],
            height=input_image_torch.shape[2],
            device=TorchDevice.choose_torch_device(),
            seed=seed,
            downsampling_factor=LATENT_SCALE_FACTOR,
            use_cpu=True,
        )

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
            pipeline =  DenoiseLatentsInvocation.create_pipeline(unet=unet, scheduler=scheduler)
            for latent_tile in latent_tiles:



        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
