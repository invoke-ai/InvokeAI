import torch

from invokeai.backend.tiled_refinement.refiners.base_refiner import BaseRefiner
from invokeai.backend.tiled_refinement.tilers.base_tiler import BaseTiler


class TiledRefiner:
    def __init__(self, tiler: BaseTiler, refiner: BaseRefiner):
        self._tiler = tiler
        self._refiner = refiner

    def refine_image(self, image: torch.Tensor) -> torch.Tensor:
        self._tiler.initialize(image)

        for i in range(self._tiler.get_num_tiles()):
            tile = self._tiler.read_tile(i)
            refined_tile_image = self._refiner.refine(tile)
            self._tiler.write_tile(refined_tile_image, i)

        return self._tiler.get_output()
