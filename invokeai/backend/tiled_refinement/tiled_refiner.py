import torch

from invokeai.backend.tiled_refinement.refiners.base_refiner import BaseRefiner
from invokeai.backend.tiled_refinement.tile import Tile
from invokeai.backend.tiled_refinement.tilers.base_tiler import BaseTiler


class TiledRefiner:
    def __init__(self, tiler: BaseTiler, refiner: BaseRefiner):
        self._tiler = tiler
        self._refiner = refiner

    def refine_image(self, image: torch.Tensor) -> torch.Tensor:
        if self._tiler.expects_overwrite():
            # Write to original as tiles are processed.
            raise NotImplementedError()
        else:
            # Write to output as tiles are processed.
            output_image = torch.zeros_like(image)

        for i in range(self._tiler.get_num_tiles(image)):
            tile = self._tiler.get_tile(image, i)

            refined_tile_image = self._refiner.refine(tile)
            self._write_tile_to_image(refined_tile_image, output_image, tile)

        return output_image

    def _write_tile_to_image(self, src_tile_image: torch.Tensor, dst_image: torch.Tensor, tile: Tile):
        assert src_tile_image.shape[-2:] == tile.image.shape[-2:]
        height, width = tile.image.shape[-2:]
        dst_image[..., tile.coords.y : tile.coords.y + height, tile.coords.x : tile.coords.x + width] = src_tile_image
