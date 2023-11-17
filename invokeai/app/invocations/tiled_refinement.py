import torch
from diffusers.image_processor import VaeImageProcessor
from PIL.Image import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
    invocation,
)
from invokeai.app.invocations.model import UNetField, VaeField
from invokeai.app.invocations.primitives import ConditioningField, ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.tiled_refinement.refiners.image_to_image_refiner import ImageToImageRefiner
from invokeai.backend.tiled_refinement.tiled_refiner import TiledRefiner
from invokeai.backend.tiled_refinement.tilers.linear_overlap_tiler import LinearOverlapTiler


@invocation(
    "tiled_refinement",
    title="Tiled Refinement",
    tags=["upscale"],
    category="upscale",
    version="1.0.0",
)
class TiledRefinementInvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Refine a high-res image by applying conditioned image-to-image operations on image tiles.

    This invocation is typically used to refine an image that has been upscaled to a higher resolution.
    """

    image: ImageField = InputField(description="The image to refine.")
    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection, ui_order=0
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection, ui_order=1
    )
    vae: VaeField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="VAE")
    unet: UNetField = InputField(description=FieldDescriptions.unet, input=Input.Connection, title="UNet")
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)

    # TODO(ryand): Add titles, descriptions.
    tile_dimension_x: int = InputField(default=512, ge=0, multiple_of=8)
    tile_dimension_y: int = InputField(default=512, ge=0, multiple_of=8)
    read_overlap: int = InputField(default=64, ge=0, multiple_of=8)
    write_overlap: int = InputField(default=64, ge=0, multiple_of=8)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        tiled_refiner = TiledRefiner(
            tiler=LinearOverlapTiler(
                tile_dimension_x=self.tile_dimension_x,
                tile_dimension_y=self.tile_dimension_y,
                read_overlap=self.read_overlap,
                write_overlap=self.write_overlap,
            ),
            refiner=ImageToImageRefiner(
                context=context,
                positive_conditioning=self.positive_conditioning,
                negative_conditioning=self.negative_conditioning,
                vae=self.vae,
                unet=self.unet,
                denoising_start=self.denoising_start,
                denoising_end=self.denoising_end,
            ),
        )

        in_pil_image = context.services.images.get_pil_image(self.image.image_name)
        in_torch_image = self._pil_to_torch_image(in_pil_image)
        out_torch_image = tiled_refiner.refine_image(in_torch_image)
        out_pil_image = self._torch_to_pil_image(out_torch_image)

        image_dto = context.services.images.create(
            image=out_pil_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )

    def _pil_to_torch_image(self, pil_image: Image) -> torch.Tensor:
        # TODO(ryand): This logic (and the logic for converting Tensor to PIL) was copied from
        # 'ImageToLatentsInvocation' (and 'LatentsToImageInvocation'). It should be organized and consolidated in one
        # place.
        torch_image = image_resized_to_grid_as_tensor(pil_image.convert("RGB"))
        if torch_image.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            torch_image = torch_image.unsqueeze(dim=0)
        return torch_image

    def _torch_to_pil_image(self, tensor_image: torch.Tensor) -> Image:
        tensor_image = (tensor_image / 2 + 0.5).clamp(0, 1)  # denormalize
        # (N, C, H, W) -> (N, H, W, C)
        np_image = tensor_image.cpu().permute(0, 2, 3, 1).float().numpy()
        # While there is a batch dimension, we currently assume that it is a batch of size 1.
        return VaeImageProcessor.numpy_to_pil(np_image)[0]
