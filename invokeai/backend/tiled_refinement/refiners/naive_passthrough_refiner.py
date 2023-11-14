import torch

from invokeai.backend.tiled_refinement.refiners.base_refiner import BaseRefiner
from invokeai.backend.tiled_refinement.tile import Tile


class NaivePassthroughRefiner(BaseRefiner):
    """A naive refiner that passes the input tile through unchanged. Typically only used for testing purposes."""

    def __init__(self):
        super().__init__()

    def refine(self, image_tile: Tile) -> torch.Tensor:
        return image_tile.image
