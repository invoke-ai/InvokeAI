import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    UIType,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap, merge_tiles_with_linear_blending
from invokeai.backend.tiles.utils import TBLR, Tile


@invocation("spandrel_image_to_image", title="Image-to-Image", tags=["upscale"], category="upscale", version="1.1.0")
class SpandrelImageToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run any spandrel image-to-image model (https://github.com/chaiNNer-org/spandrel)."""

    image: ImageField = InputField(description="The input image")
    image_to_image_model: ModelIdentifierField = InputField(
        title="Image-to-Image Model",
        description=FieldDescriptions.spandrel_image_to_image_model,
        ui_type=UIType.SpandrelImageToImageModel,
    )
    tile_size: int = InputField(
        default=512, description="The tile size for tiled image-to-image. Set to 0 to disable tiling."
    )

    def _scale_tile(self, tile: Tile, scale: int) -> Tile:
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

    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Images are converted to RGB, because most models don't support an alpha channel. In the future, we may want to
        # revisit this.
        image = context.images.get_pil(self.image.image_name, mode="RGB")

        # Compute the image tiles.
        if self.tile_size > 0:
            min_overlap = 20
            tiles = calc_tiles_min_overlap(
                image_height=image.height,
                image_width=image.width,
                tile_height=self.tile_size,
                tile_width=self.tile_size,
                min_overlap=min_overlap,
            )
        else:
            # No tiling. Generate a single tile that covers the entire image.
            min_overlap = 0
            tiles = [
                Tile(
                    coords=TBLR(top=0, bottom=image.height, left=0, right=image.width),
                    overlap=TBLR(top=0, bottom=0, left=0, right=0),
                )
            ]

        # Prepare input image for inference.
        image_tensor = SpandrelImageToImageModel.pil_to_tensor(image)

        # Load the model.
        spandrel_model_info = context.models.load(self.image_to_image_model)

        # Run the model on each tile.
        output_tiles: list[torch.Tensor] = []
        scale: int = 1
        with spandrel_model_info as spandrel_model:
            assert isinstance(spandrel_model, SpandrelImageToImageModel)

            # Scale the tiles for re-assembling the final image.
            scale = spandrel_model.scale
            scaled_tiles = [self._scale_tile(tile, scale=scale) for tile in tiles]

            image_tensor = image_tensor.to(device=spandrel_model.device, dtype=spandrel_model.dtype)

            for tile in tqdm(tiles, desc="Upscaling Tiles"):
                output_tile = spandrel_model.run(
                    image_tensor[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right]
                )
                output_tiles.append(output_tile)

        # Merge tiles into output image.
        np_output_tiles = [np.array(SpandrelImageToImageModel.tensor_to_pil(tile)) for tile in output_tiles]
        _, channels, height, width = image_tensor.shape
        np_out_image = np.zeros((height * scale, width * scale, channels), dtype=np.uint8)
        merge_tiles_with_linear_blending(
            dst_image=np_out_image, tiles=scaled_tiles, tile_images=np_output_tiles, blend_amount=min_overlap // 2
        )

        # Convert the output tensor to a PIL image.
        pil_image = Image.fromarray(np_out_image)
        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)
