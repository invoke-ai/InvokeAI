import torch

from invokeai.backend.tiled_refinement.refiners.base_refiner import BaseRefiner


class NaivePassthroughRefiner(BaseRefiner):
    """A naive refiner that passes the input tile through unchanged. Typically only used for testing purposes."""

    def __init__(self):
        super().__init__()

    def refine(self, image_tile: torch.Tensor) -> torch.Tensor:
        return image_tile
