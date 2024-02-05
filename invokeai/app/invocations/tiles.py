from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import ImageField, Input, InputField, OutputField, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.tiles.tiles import (
    calc_tiles_even_split,
    calc_tiles_min_overlap,
    calc_tiles_with_overlap,
    merge_tiles_with_linear_blending,
    merge_tiles_with_seam_blending,
)
from invokeai.backend.tiles.utils import Tile


class TileWithImage(BaseModel):
    tile: Tile
    image: ImageField


@invocation_output("calculate_image_tiles_output")
class CalculateImageTilesOutput(BaseInvocationOutput):
    tiles: list[Tile] = OutputField(description="The tiles coordinates that cover a particular image shape.")


@invocation(
    "calculate_image_tiles",
    title="Calculate Image Tiles",
    tags=["tiles"],
    category="tiles",
    version="1.0.0",
    classification=Classification.Beta,
)
class CalculateImageTilesInvocation(BaseInvocation):
    """Calculate the coordinates and overlaps of tiles that cover a target image shape."""

    image_width: int = InputField(ge=1, default=1024, description="The image width, in pixels, to calculate tiles for.")
    image_height: int = InputField(
        ge=1, default=1024, description="The image height, in pixels, to calculate tiles for."
    )
    tile_width: int = InputField(ge=1, default=576, description="The tile width, in pixels.")
    tile_height: int = InputField(ge=1, default=576, description="The tile height, in pixels.")
    overlap: int = InputField(
        ge=0,
        default=128,
        description="The target overlap, in pixels, between adjacent tiles. Adjacent tiles will overlap by at least this amount",
    )

    def invoke(self, context: InvocationContext) -> CalculateImageTilesOutput:
        tiles = calc_tiles_with_overlap(
            image_height=self.image_height,
            image_width=self.image_width,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            overlap=self.overlap,
        )
        return CalculateImageTilesOutput(tiles=tiles)


@invocation(
    "calculate_image_tiles_even_split",
    title="Calculate Image Tiles Even Split",
    tags=["tiles"],
    category="tiles",
    version="1.1.0",
    classification=Classification.Beta,
)
class CalculateImageTilesEvenSplitInvocation(BaseInvocation):
    """Calculate the coordinates and overlaps of tiles that cover a target image shape."""

    image_width: int = InputField(ge=1, default=1024, description="The image width, in pixels, to calculate tiles for.")
    image_height: int = InputField(
        ge=1, default=1024, description="The image height, in pixels, to calculate tiles for."
    )
    num_tiles_x: int = InputField(
        default=2,
        ge=1,
        description="Number of tiles to divide image into on the x axis",
    )
    num_tiles_y: int = InputField(
        default=2,
        ge=1,
        description="Number of tiles to divide image into on the y axis",
    )
    overlap: int = InputField(
        default=128,
        ge=0,
        multiple_of=8,
        description="The overlap, in pixels, between adjacent tiles.",
    )

    def invoke(self, context: InvocationContext) -> CalculateImageTilesOutput:
        tiles = calc_tiles_even_split(
            image_height=self.image_height,
            image_width=self.image_width,
            num_tiles_x=self.num_tiles_x,
            num_tiles_y=self.num_tiles_y,
            overlap=self.overlap,
        )
        return CalculateImageTilesOutput(tiles=tiles)


@invocation(
    "calculate_image_tiles_min_overlap",
    title="Calculate Image Tiles Minimum Overlap",
    tags=["tiles"],
    category="tiles",
    version="1.0.0",
    classification=Classification.Beta,
)
class CalculateImageTilesMinimumOverlapInvocation(BaseInvocation):
    """Calculate the coordinates and overlaps of tiles that cover a target image shape."""

    image_width: int = InputField(ge=1, default=1024, description="The image width, in pixels, to calculate tiles for.")
    image_height: int = InputField(
        ge=1, default=1024, description="The image height, in pixels, to calculate tiles for."
    )
    tile_width: int = InputField(ge=1, default=576, description="The tile width, in pixels.")
    tile_height: int = InputField(ge=1, default=576, description="The tile height, in pixels.")
    min_overlap: int = InputField(default=128, ge=0, description="Minimum overlap between adjacent tiles, in pixels.")

    def invoke(self, context: InvocationContext) -> CalculateImageTilesOutput:
        tiles = calc_tiles_min_overlap(
            image_height=self.image_height,
            image_width=self.image_width,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            min_overlap=self.min_overlap,
        )
        return CalculateImageTilesOutput(tiles=tiles)


@invocation_output("tile_to_properties_output")
class TileToPropertiesOutput(BaseInvocationOutput):
    coords_left: int = OutputField(description="Left coordinate of the tile relative to its parent image.")
    coords_right: int = OutputField(description="Right coordinate of the tile relative to its parent image.")
    coords_top: int = OutputField(description="Top coordinate of the tile relative to its parent image.")
    coords_bottom: int = OutputField(description="Bottom coordinate of the tile relative to its parent image.")

    # HACK: The width and height fields are 'meta' fields that can easily be calculated from the other fields on this
    # object. Including redundant fields that can cheaply/easily be re-calculated goes against conventional API design
    # principles. These fields are included, because 1) they are often useful in tiled workflows, and 2) they are
    # difficult to calculate in a workflow (even though it's just a couple of subtraction nodes the graph gets
    # surprisingly complicated).
    width: int = OutputField(description="The width of the tile. Equal to coords_right - coords_left.")
    height: int = OutputField(description="The height of the tile. Equal to coords_bottom - coords_top.")

    overlap_top: int = OutputField(description="Overlap between this tile and its top neighbor.")
    overlap_bottom: int = OutputField(description="Overlap between this tile and its bottom neighbor.")
    overlap_left: int = OutputField(description="Overlap between this tile and its left neighbor.")
    overlap_right: int = OutputField(description="Overlap between this tile and its right neighbor.")


@invocation(
    "tile_to_properties",
    title="Tile to Properties",
    tags=["tiles"],
    category="tiles",
    version="1.0.0",
    classification=Classification.Beta,
)
class TileToPropertiesInvocation(BaseInvocation):
    """Split a Tile into its individual properties."""

    tile: Tile = InputField(description="The tile to split into properties.")

    def invoke(self, context: InvocationContext) -> TileToPropertiesOutput:
        return TileToPropertiesOutput(
            coords_left=self.tile.coords.left,
            coords_right=self.tile.coords.right,
            coords_top=self.tile.coords.top,
            coords_bottom=self.tile.coords.bottom,
            width=self.tile.coords.right - self.tile.coords.left,
            height=self.tile.coords.bottom - self.tile.coords.top,
            overlap_top=self.tile.overlap.top,
            overlap_bottom=self.tile.overlap.bottom,
            overlap_left=self.tile.overlap.left,
            overlap_right=self.tile.overlap.right,
        )


@invocation_output("pair_tile_image_output")
class PairTileImageOutput(BaseInvocationOutput):
    tile_with_image: TileWithImage = OutputField(description="A tile description with its corresponding image.")


@invocation(
    "pair_tile_image",
    title="Pair Tile with Image",
    tags=["tiles"],
    category="tiles",
    version="1.0.0",
    classification=Classification.Beta,
)
class PairTileImageInvocation(BaseInvocation):
    """Pair an image with its tile properties."""

    # TODO(ryand): The only reason that PairTileImage is needed is because the iterate/collect nodes don't preserve
    # order. Can this be fixed?

    image: ImageField = InputField(description="The tile image.")
    tile: Tile = InputField(description="The tile properties.")

    def invoke(self, context: InvocationContext) -> PairTileImageOutput:
        return PairTileImageOutput(
            tile_with_image=TileWithImage(
                tile=self.tile,
                image=self.image,
            )
        )


BLEND_MODES = Literal["Linear", "Seam"]


@invocation(
    "merge_tiles_to_image",
    title="Merge Tiles to Image",
    tags=["tiles"],
    category="tiles",
    version="1.1.0",
    classification=Classification.Beta,
)
class MergeTilesToImageInvocation(BaseInvocation, WithMetadata):
    """Merge multiple tile images into a single image."""

    # Inputs
    tiles_with_images: list[TileWithImage] = InputField(description="A list of tile images with tile properties.")
    blend_mode: BLEND_MODES = InputField(
        default="Seam",
        description="blending type Linear or Seam",
        input=Input.Direct,
    )
    blend_amount: int = InputField(
        default=32,
        ge=0,
        description="The amount to blend adjacent tiles in pixels. Must be <= the amount of overlap between adjacent tiles.",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        images = [twi.image for twi in self.tiles_with_images]
        tiles = [twi.tile for twi in self.tiles_with_images]

        # Infer the output image dimensions from the max/min tile limits.
        height = 0
        width = 0
        for tile in tiles:
            height = max(height, tile.coords.bottom)
            width = max(width, tile.coords.right)

        # Get all tile images for processing.
        # TODO(ryand): It pains me that we spend time PNG decoding each tile from disk when they almost certainly
        # existed in memory at an earlier point in the graph.
        tile_np_images: list[np.ndarray] = []
        for image in images:
            pil_image = context.images.get_pil(image.image_name)
            pil_image = pil_image.convert("RGB")
            tile_np_images.append(np.array(pil_image))

        # Prepare the output image buffer.
        # Check the first tile to determine how many image channels are expected in the output.
        channels = tile_np_images[0].shape[-1]
        dtype = tile_np_images[0].dtype
        np_image = np.zeros(shape=(height, width, channels), dtype=dtype)
        if self.blend_mode == "Linear":
            merge_tiles_with_linear_blending(
                dst_image=np_image, tiles=tiles, tile_images=tile_np_images, blend_amount=self.blend_amount
            )
        elif self.blend_mode == "Seam":
            merge_tiles_with_seam_blending(
                dst_image=np_image, tiles=tiles, tile_images=tile_np_images, blend_amount=self.blend_amount
            )
        else:
            raise ValueError(f"Unsupported blend mode: '{self.blend_mode}'.")

        # Convert into a PIL image and save
        pil_image = Image.fromarray(np_image)

        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)
