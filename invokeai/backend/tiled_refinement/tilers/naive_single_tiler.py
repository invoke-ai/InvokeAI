import torch

from invokeai.backend.tiled_refinement.tilers.base_tiler import BaseTiler, TilerNotInitializedError


class NaiveSingleTiler(BaseTiler):
    """A naive tiler that produces a single tile containing the entire image. Typically only used for testing
    purposes.
    """

    def __init__(self):
        super().__init__()

        self._image = None
        self._out_image = None

    def initialize(self, image: torch.Tensor):
        self._image = image
        self._out_image = image.clone()

    def get_num_tiles(self) -> int:
        if self._image is None:
            raise TilerNotInitializedError("Called get_num_tiles() before calling initialize(...).")
        return 1

    def read_tile(self, i: int) -> torch.Tensor:
        return self._image.clone()

    def write_tile(self, tile_image: torch.Tensor, i: int):
        self._out_image = tile_image

    def get_output(self) -> torch.Tensor:
        return self._out_image
