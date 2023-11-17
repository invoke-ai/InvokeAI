import numpy as np
from PIL import Image
from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    WithMetadata,
    WithWorkflow,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.backend.tiles.tiles import calc_tiles, merge_tiles_with_linear_blending
from invokeai.backend.tiles.utils import Tile

# TODO(ryand): Is this important?
_DIMENSION_MULTIPLE_OF = 8


class TileWithImage(BaseModel):
    tile: Tile
    image: ImageField


@invocation_output("calc_tiles_output")
class CalcTilesOutput(BaseInvocationOutput):
    # TODO(ryand): Add description from FieldDescriptions.
    tiles: list[Tile] = OutputField(description="")


@invocation("calculate_tiles", title="Calculate Tiles", tags=["tiles"], category="tiles", version="1.0.0")
class CalcTiles(BaseInvocation):
    """TODO(ryand)"""

    # Inputs
    image_height: int = InputField(ge=1)
    image_width: int = InputField(ge=1)
    tile_height: int = InputField(ge=1, multiple_of=_DIMENSION_MULTIPLE_OF, default=576)
    tile_width: int = InputField(ge=1, multiple_of=_DIMENSION_MULTIPLE_OF, default=576)
    overlap: int = InputField(ge=0, multiple_of=_DIMENSION_MULTIPLE_OF, default=64)

    def invoke(self, context: InvocationContext) -> CalcTilesOutput:
        tiles = calc_tiles(
            image_height=self.image_height,
            image_width=self.image_width,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            overlap=self.overlap,
        )
        return CalcTilesOutput(tiles=tiles)


@invocation_output("tile_to_properties_output")
class TileToPropertiesOutput(BaseInvocationOutput):
    # TODO(ryand): Add descriptions.
    coords_top: int = OutputField(description="")
    coords_bottom: int = OutputField(description="")
    coords_left: int = OutputField(description="")
    coords_right: int = OutputField(description="")

    overlap_top: int = OutputField(description="")
    overlap_bottom: int = OutputField(description="")
    overlap_left: int = OutputField(description="")
    overlap_right: int = OutputField(description="")


@invocation("tile_to_properties")
class TileToProperties(BaseInvocation):
    """Split a Tile into its individual properties."""

    tile: Tile = InputField()

    def invoke(self, context: InvocationContext) -> TileToPropertiesOutput:
        return TileToPropertiesOutput(
            coords_top=self.tile.coords.top,
            coords_bottom=self.tile.coords.bottom,
            coords_left=self.tile.coords.left,
            coords_right=self.tile.coords.right,
            overlap_top=self.tile.overlap.top,
            overlap_bottom=self.tile.overlap.bottom,
            overlap_left=self.tile.overlap.left,
            overlap_right=self.tile.overlap.right,
        )


# HACK(ryand): The only reason that PairTileImage is needed is because the iterate/collect nodes don't preserve order.
# Can this be fixed?


@invocation_output("pair_tile_image_output")
class PairTileImageOutput(BaseInvocationOutput):
    tile_with_image: TileWithImage = OutputField(description="")


@invocation("pair_tile_image", title="Pair Tile with Image", tags=["tiles"], category="tiles", version="1.0.0")
class PairTileImage(BaseInvocation):
    image: ImageField = InputField()
    tile: Tile = InputField()

    def invoke(self, context: InvocationContext) -> PairTileImageOutput:
        return PairTileImageOutput(
            tile_with_image=TileWithImage(
                tile=self.tile,
                image=self.image,
            )
        )


@invocation("merge_tiles_to_image", title="Merge Tiles To Image", tags=["tiles"], category="tiles", version="1.0.0")
class MergeTilesToImage(BaseInvocation, WithMetadata, WithWorkflow):
    """TODO(ryand)"""

    # Inputs
    image_height: int = InputField(ge=1)
    image_width: int = InputField(ge=1)
    tiles_with_images: list[TileWithImage] = InputField()
    blend_amount: int = InputField(ge=0)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        images = [twi.image for twi in self.tiles_with_images]
        tiles = [twi.tile for twi in self.tiles_with_images]

        # Get all tile images for processing.
        # TODO(ryand): It pains me that we spend time PNG decoding each tile from disk when they almost certainly
        # existed in memory at an earlier point in the graph.
        tile_np_images: list[np.ndarray] = []
        for image in images:
            pil_image = context.services.images.get_pil_image(image.image_name)
            pil_image = pil_image.convert("RGB")
            tile_np_images.append(np.array(pil_image))

        # Prepare the output image buffer.
        # Check the first tile to determine how many image channels are expected in the output.
        channels = tile_np_images[0].shape[-1]
        dtype = tile_np_images[0].dtype
        np_image = np.zeros(shape=(self.image_height, self.image_width, channels), dtype=dtype)

        merge_tiles_with_linear_blending(
            dst_image=np_image, tiles=tiles, tile_images=tile_np_images, blend_amount=self.blend_amount
        )
        pil_image = Image.fromarray(np_image)

        image_dto = context.services.images.create(
            image=pil_image,
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
