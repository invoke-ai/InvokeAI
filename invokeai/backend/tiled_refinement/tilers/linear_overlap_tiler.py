import math
from dataclasses import dataclass
from typing import Optional

import torch

from invokeai.backend.tiled_refinement.tilers.base_tiler import BaseTiler, TilerNotInitializedError
from invokeai.backend.tiled_refinement.tilers.utils import TBLR, crop


@dataclass
class TileProperties:
    # Tile coordinates.
    coords: TBLR

    # Tile read overlap with neighbors.
    # E.g. 'overlap.top' is the number of pixels of overlap between this tile and its top neighbor.
    overlap: TBLR


class LinearOverlapTiler(BaseTiler):
    """A basic tiler."""

    def __init__(
        self,
        tile_dimension_x: int,
        tile_dimension_y: int,
        read_overlap_x: int,
        read_overlap_y: int,
        write_blend_x: int,
        write_blend_y: int,
    ):
        super().__init__()
        self._tile_dimension_x = tile_dimension_x
        self._tile_dimension_y = tile_dimension_y
        self._read_overlap_x = read_overlap_x
        self._read_overlap_y = read_overlap_y
        self._write_blend_x = write_blend_x
        self._write_blend_y = write_blend_y

        self._image: Optional[torch.Tensor] = None
        self._out_image: Optional[torch.Tensor] = None
        # Pre-computed tiles stored after calling initialize(...).
        self._tile_props: Optional[list[TileProperties]] = None

        # Validate inputs.
        assert self._tile_dimension_x > self._read_overlap_x
        assert self._tile_dimension_y > self._read_overlap_y
        assert self._read_overlap_x >= self._write_blend_x
        assert self._read_overlap_y >= self._write_blend_y

    def initialize(self, image: torch.Tensor):
        image_height, image_width = image.shape[-2:]
        assert image_height >= self._tile_dimension_y
        assert image_width >= self._tile_dimension_x

        non_overlap_per_tile_x = self._tile_dimension_x - self._read_overlap_x
        non_overlap_per_tile_y = self._tile_dimension_y - self._read_overlap_y

        num_tiles_x = math.ceil((image_width - self._read_overlap_x) / non_overlap_per_tile_x)
        num_tiles_y = math.ceil((image_width - self._read_overlap_y) / non_overlap_per_tile_y)

        # Calculate tile coordinates and overlaps.
        tiles: list[TileProperties] = []
        for tile_idx_y in range(num_tiles_y):
            for tile_idx_x in range(num_tiles_x):
                tile = TileProperties(
                    coords=TBLR(
                        top=tile_idx_y * non_overlap_per_tile_y,
                        bottom=tile_idx_y * non_overlap_per_tile_y + self._tile_dimension_y,
                        left=tile_idx_x * non_overlap_per_tile_x,
                        right=tile_idx_x * non_overlap_per_tile_x + self._tile_dimension_x,
                    ),
                    overlap=TBLR(
                        top=0 if tile_idx_y == 0 else self._read_overlap_y,
                        bottom=self._read_overlap_y,
                        left=0 if tile_idx_x == 0 else self._read_overlap_x,
                        right=self._read_overlap_x,
                    ),
                )

                if tile.coords.bottom > image_height:
                    # If this tile would go off the bottom of the image, shift it so that it is aligned with the bottom
                    # of the image.
                    tile.coords.bottom = image_height
                    tile.coords.top = image_height - self._tile_dimension_y
                    tile.overlap.bottom = 0
                    # Note that this could result in a large overlap between this tile and the one above it.
                    top_neighbor_bottom = (tile_idx_y - 1) * non_overlap_per_tile_y + self._tile_dimension_y
                    tile.overlap.top = top_neighbor_bottom - tile.coords.top

                if tile.coords.right > image_width:
                    # If this tile would go off the right edge of the image, shift it so that it is aligned with the
                    # right edge of the image.
                    tile.coords.right = image_width
                    tile.coords.left = image_width - self._tile_dimension_x
                    tile.overlap.right = 0
                    # Note that this could result in a large overlap between this tile and the one to its left.
                    left_neighbor_right = (tile_idx_x - 1) * non_overlap_per_tile_x + self._tile_dimension_x
                    tile.overlap.left = left_neighbor_right - tile.coords.left

                tiles.append(tile)

        self._image = image
        self._out_image = torch.zeros_like(self._image)
        self._tile_props = tiles

    def get_num_tiles(self) -> int:
        if self._tile_props is None:
            raise TilerNotInitializedError("Called get_num_tiles() before calling initialize(...).")
        return len(self._tile_props)

    def read_tile(self, i: int) -> torch.Tensor:
        if self._tile_props is None or self._image is None:
            raise TilerNotInitializedError("Called read_tile() before calling initialize(...).")

        return crop(self._image, self._tile_props[i].coords)

    def write_tile(self, tile_image: torch.Tensor, i: int):
        raise NotImplementedError()

    def get_output(self) -> torch.Tensor:
        raise NotImplementedError()
