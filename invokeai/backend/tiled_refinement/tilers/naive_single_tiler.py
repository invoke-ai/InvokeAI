import torch

from invokeai.backend.tiled_refinement.tile import Tile, XYCoords
from invokeai.backend.tiled_refinement.tilers.base_tiler import BaseTiler


class NaiveSingleTiler(BaseTiler):
    """A naive tiler that produces a single tile containing the entire image. Typically only used for testing
    purposes.
    """

    def __init__(self):
        super().__init__()

    def get_num_tiles(self, image: torch.Tensor) -> int:
        return 1

    def get_tile(self, image: torch.Tensor, i: int) -> Tile:
        # TODO(ryand): Revisit this masking format.
        mask = torch.ones_like(image, dtype=torch.uint8)
        return Tile(
            image=image,
            coords=XYCoords(x=0, y=0),
            refinement_mask=mask,
            write_mask=mask,
        )

    def expects_overwrite(self) -> bool:
        return False
