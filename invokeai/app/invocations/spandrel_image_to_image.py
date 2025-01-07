import functools
from typing import Callable

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
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.tiles.utils import TBLR, Tile
from invokeai.backend.util.devices import TorchDevice


@invocation("spandrel_image_to_image", title="Image-to-Image", tags=["upscale"], category="upscale", version="1.3.0")
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

    @classmethod
    def scale_tile(cls, tile: Tile, scale: int) -> Tile:
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

    @classmethod
    def upscale_image(
        cls,
        image: Image.Image,
        tile_size: int,
        spandrel_model: SpandrelImageToImageModel,
        is_canceled: Callable[[], bool],
        step_callback: Callable[[int, int], None],
    ) -> Image.Image:
        # Compute the image tiles.
        if tile_size > 0:
            min_overlap = 20
            tiles = calc_tiles_min_overlap(
                image_height=image.height,
                image_width=image.width,
                tile_height=tile_size,
                tile_width=tile_size,
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

        # Sort tiles first by left x coordinate, then by top y coordinate. During tile processing, we want to iterate
        # over tiles left-to-right, top-to-bottom.
        tiles = sorted(tiles, key=lambda x: x.coords.left)
        tiles = sorted(tiles, key=lambda x: x.coords.top)

        # Prepare input image for inference.
        image_tensor = SpandrelImageToImageModel.pil_to_tensor(image)

        # Scale the tiles for re-assembling the final image.
        scale = spandrel_model.scale
        scaled_tiles = [cls.scale_tile(tile, scale=scale) for tile in tiles]

        # Prepare the output tensor.
        _, channels, height, width = image_tensor.shape
        output_tensor = torch.zeros(
            (height * scale, width * scale, channels), dtype=torch.uint8, device=torch.device("cpu")
        )

        image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=spandrel_model.dtype)

        # Run the model on each tile.
        pbar = tqdm(list(zip(tiles, scaled_tiles, strict=True)), desc="Upscaling Tiles")

        # Update progress, starting with 0.
        step_callback(0, pbar.total)

        for tile, scaled_tile in pbar:
            # Exit early if the invocation has been canceled.
            if is_canceled():
                raise CanceledException

            # Extract the current tile from the input tensor.
            input_tile = image_tensor[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right]

            # Run the model on the tile.
            output_tile = spandrel_model.run(input_tile)

            # Convert the output tile into the output tensor's format.
            # (N, C, H, W) -> (C, H, W)
            output_tile = output_tile.squeeze(0)
            # (C, H, W) -> (H, W, C)
            output_tile = output_tile.permute(1, 2, 0)
            output_tile = output_tile.clamp(0, 1)
            output_tile = (output_tile * 255).to(dtype=torch.uint8, device=torch.device("cpu"))

            # Merge the output tile into the output tensor.
            # We only keep half of the overlap on the top and left side of the tile. We do this in case there are
            # edge artifacts. We don't bother with any 'blending' in the current implementation - for most upscalers
            # it seems unnecessary, but we may find a need in the future.
            top_overlap = scaled_tile.overlap.top // 2
            left_overlap = scaled_tile.overlap.left // 2
            output_tensor[
                scaled_tile.coords.top + top_overlap : scaled_tile.coords.bottom,
                scaled_tile.coords.left + left_overlap : scaled_tile.coords.right,
                :,
            ] = output_tile[top_overlap:, left_overlap:, :]

            step_callback(pbar.n + 1, pbar.total)

        # Convert the output tensor to a PIL image.
        np_image = output_tensor.detach().numpy().astype(np.uint8)
        pil_image = Image.fromarray(np_image)

        return pil_image

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Images are converted to RGB, because most models don't support an alpha channel. In the future, we may want to
        # revisit this.
        image = context.images.get_pil(self.image.image_name, mode="RGB")

        def step_callback(step: int, total_steps: int) -> None:
            context.util.signal_progress(
                message=f"Processing tile {step}/{total_steps}",
                percentage=step / total_steps,
            )

        # Do the upscaling.
        with context.models.load(self.image_to_image_model) as spandrel_model:
            assert isinstance(spandrel_model, SpandrelImageToImageModel)

            # Upscale the image
            pil_image = self.upscale_image(
                image, self.tile_size, spandrel_model, context.util.is_canceled, step_callback
            )

        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)


@invocation(
    "spandrel_image_to_image_autoscale",
    title="Image-to-Image (Autoscale)",
    tags=["upscale"],
    category="upscale",
    version="1.0.0",
)
class SpandrelImageToImageAutoscaleInvocation(SpandrelImageToImageInvocation):
    """Run any spandrel image-to-image model (https://github.com/chaiNNer-org/spandrel) until the target scale is reached."""

    scale: float = InputField(
        default=4.0,
        gt=0.0,
        le=16.0,
        description="The final scale of the output image. If the model does not upscale the image, this will be ignored.",
    )
    fit_to_multiple_of_8: bool = InputField(
        default=False,
        description="If true, the output image will be resized to the nearest multiple of 8 in both dimensions.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Images are converted to RGB, because most models don't support an alpha channel. In the future, we may want to
        # revisit this.
        image = context.images.get_pil(self.image.image_name, mode="RGB")

        # The target size of the image, determined by the provided scale. We'll run the upscaler until we hit this size.
        # Later, we may mutate this value if the model doesn't upscale the image or if the user requested a multiple of 8.
        target_width = int(image.width * self.scale)
        target_height = int(image.height * self.scale)

        def step_callback(iteration: int, step: int, total_steps: int) -> None:
            context.util.signal_progress(
                message=self._get_progress_message(iteration, step, total_steps),
                percentage=step / total_steps,
            )

        # Do the upscaling.
        with context.models.load(self.image_to_image_model) as spandrel_model:
            assert isinstance(spandrel_model, SpandrelImageToImageModel)

            iteration = 1
            context.util.signal_progress(self._get_progress_message(iteration))

            # First pass of upscaling. Note: `pil_image` will be mutated.
            pil_image = self.upscale_image(
                image,
                self.tile_size,
                spandrel_model,
                context.util.is_canceled,
                functools.partial(step_callback, iteration),
            )

            # Some models don't upscale the image, but we have no way to know this in advance. We'll check if the model
            # upscaled the image and run the loop below if it did. We'll require the model to upscale both dimensions
            # to be considered an upscale model.
            is_upscale_model = pil_image.width > image.width and pil_image.height > image.height

            if is_upscale_model:
                # This is an upscale model, so we should keep upscaling until we reach the target size.
                while pil_image.width < target_width or pil_image.height < target_height:
                    iteration += 1
                    context.util.signal_progress(self._get_progress_message(iteration))
                    pil_image = self.upscale_image(
                        pil_image,
                        self.tile_size,
                        spandrel_model,
                        context.util.is_canceled,
                        functools.partial(step_callback, iteration),
                    )

                    # Sanity check to prevent excessive or infinite loops. All known upscaling models are at least 2x.
                    # Our max scale is 16x, so with a 2x model, we should never exceed 16x == 2^4 -> 4 iterations.
                    # We'll allow one extra iteration "just in case" and bail at 5 upscaling iterations. In practice,
                    # we should never reach this limit.
                    if iteration >= 5:
                        context.logger.warning(
                            "Upscale loop reached maximum iteration count of 5, stopping upscaling early."
                        )
                        break
            else:
                # This model doesn't upscale the image. We should ignore the scale parameter, modifying the output size
                # to be the same as the processed image size.

                # The output size is now the size of the processed image.
                target_width = pil_image.width
                target_height = pil_image.height

                # Warn the user if they requested a scale greater than 1.
                if self.scale > 1:
                    context.logger.warning(
                        "Model does not increase the size of the image, but a greater scale than 1 was requested. Image will not be scaled."
                    )

        # We may need to resize the image to a multiple of 8. Use floor division to ensure we don't scale the image up
        # in the final resize
        if self.fit_to_multiple_of_8:
            target_width = int(target_width // 8 * 8)
            target_height = int(target_height // 8 * 8)

        # Final resize. Per PIL documentation, Lanczos provides the best quality for both upscale and downscale.
        # See: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
        pil_image = pil_image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)

    @classmethod
    def _get_progress_message(cls, iteration: int, step: int | None = None, total_steps: int | None = None) -> str:
        if step is not None and total_steps is not None:
            return f"Processing iteration {iteration}, tile {step}/{total_steps}"

        return f"Processing iteration {iteration}"
