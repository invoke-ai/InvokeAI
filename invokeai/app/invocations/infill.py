# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from typing import Literal, Optional, get_args

import numpy as np
import math
from PIL import Image, ImageOps
from pydantic import Field

from invokeai.app.invocations.image import ImageOutput
from invokeai.app.util.misc import SEED_MAX, get_random_seed
from invokeai.backend.image_util.patchmatch import PatchMatch

from ..models.image import ColorField, ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import (
    BaseInvocation,
    InvocationConfig,
    InvocationContext,
)


def infill_methods() -> list[str]:
    methods = [
        "tile",
        "solid",
    ]
    if PatchMatch.patchmatch_available():
        methods.insert(0, "patchmatch")
    return methods


INFILL_METHODS = Literal[tuple(infill_methods())]
DEFAULT_INFILL_METHOD = "patchmatch" if "patchmatch" in get_args(INFILL_METHODS) else "tile"


def infill_patchmatch(im: Image.Image) -> Image.Image:
    if im.mode != "RGBA":
        return im

    # Skip patchmatch if patchmatch isn't available
    if not PatchMatch.patchmatch_available():
        return im

    # Patchmatch (note, we may want to expose patch_size? Increasing it significantly impacts performance though)
    im_patched_np = PatchMatch.inpaint(im.convert("RGB"), ImageOps.invert(im.split()[-1]), patch_size=3)
    im_patched = Image.fromarray(im_patched_np, mode="RGB")
    return im_patched


def get_tile_images(image: np.ndarray, width=8, height=8):
    _nrows, _ncols, depth = image.shape
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False,
    )


def tile_fill_missing(im: Image.Image, tile_size: int = 16, seed: Optional[int] = None) -> Image.Image:
    # Only fill if there's an alpha layer
    if im.mode != "RGBA":
        return im

    a = np.asarray(im, dtype=np.uint8)

    tile_size_tuple = (tile_size, tile_size)

    # Get the image as tiles of a specified size
    tiles = get_tile_images(a, *tile_size_tuple).copy()

    # Get the mask as tiles
    tiles_mask = tiles[:, :, :, :, 3]

    # Find any mask tiles with any fully transparent pixels (we will be replacing these later)
    tmask_shape = tiles_mask.shape
    tiles_mask = tiles_mask.reshape(math.prod(tiles_mask.shape))
    n, ny = (math.prod(tmask_shape[0:2])), math.prod(tmask_shape[2:])
    tiles_mask = tiles_mask > 0
    tiles_mask = tiles_mask.reshape((n, ny)).all(axis=1)

    # Get RGB tiles in single array and filter by the mask
    tshape = tiles.shape
    tiles_all = tiles.reshape((math.prod(tiles.shape[0:2]), *tiles.shape[2:]))
    filtered_tiles = tiles_all[tiles_mask]

    if len(filtered_tiles) == 0:
        return im

    # Find all invalid tiles and replace with a random valid tile
    replace_count = (tiles_mask == False).sum()
    rng = np.random.default_rng(seed=seed)
    tiles_all[np.logical_not(tiles_mask)] = filtered_tiles[rng.choice(filtered_tiles.shape[0], replace_count), :, :, :]

    # Convert back to an image
    tiles_all = tiles_all.reshape(tshape)
    tiles_all = tiles_all.swapaxes(1, 2)
    st = tiles_all.reshape(
        (
            math.prod(tiles_all.shape[0:2]),
            math.prod(tiles_all.shape[2:4]),
            tiles_all.shape[4],
        )
    )
    si = Image.fromarray(st, mode="RGBA")

    return si


class InfillColorInvocation(BaseInvocation):
    """Infills transparent areas of an image with a solid color"""

    type: Literal["infill_rgba"] = "infill_rgba"
    image: Optional[ImageField] = Field(default=None, description="The image to infill")
    color: ColorField = Field(
        default=ColorField(r=127, g=127, b=127, a=255),
        description="The color to use to infill",
    )

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Color Infill", "tags": ["image", "inpaint", "color", "infill"]},
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        solid_bg = Image.new("RGBA", image.size, self.color.tuple())
        infilled = Image.alpha_composite(solid_bg, image.convert("RGBA"))

        infilled.paste(image, (0, 0), image.split()[-1])

        image_dto = context.services.images.create(
            image=infilled,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )


class InfillTileInvocation(BaseInvocation):
    """Infills transparent areas of an image with tiles of the image"""

    type: Literal["infill_tile"] = "infill_tile"

    image: Optional[ImageField] = Field(default=None, description="The image to infill")
    tile_size: int = Field(default=32, ge=1, description="The tile size (px)")
    seed: int = Field(
        ge=0,
        le=SEED_MAX,
        description="The seed to use for tile generation (omit for random)",
        default_factory=get_random_seed,
    )

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Tile Infill", "tags": ["image", "inpaint", "tile", "infill"]},
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        infilled = tile_fill_missing(image.copy(), seed=self.seed, tile_size=self.tile_size)
        infilled.paste(image, (0, 0), image.split()[-1])

        image_dto = context.services.images.create(
            image=infilled,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )


class InfillPatchMatchInvocation(BaseInvocation):
    """Infills transparent areas of an image using the PatchMatch algorithm"""

    type: Literal["infill_patchmatch"] = "infill_patchmatch"

    image: Optional[ImageField] = Field(default=None, description="The image to infill")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Patch Match Infill", "tags": ["image", "inpaint", "patchmatch", "infill"]},
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        if PatchMatch.patchmatch_available():
            infilled = infill_patchmatch(image.copy())
        else:
            raise ValueError("PatchMatch is not available on this system")

        image_dto = context.services.images.create(
            image=infilled,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
