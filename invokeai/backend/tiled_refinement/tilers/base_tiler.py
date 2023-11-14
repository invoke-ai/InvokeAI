from abc import ABC, abstractmethod

import torch

from invokeai.backend.tiled_refinement.tile import Tile


class BaseTiler(ABC):
    @abstractmethod
    def get_num_tiles(self, image: torch.Tensor) -> int:
        """The number of tiles that will be produced by this Tiler for 'image'."""
        ...

    @abstractmethod
    def get_tile(self, image: torch.Tensor, i: int) -> Tile:
        """Return the 'i'th tile from 'image'."""
        ...

    @abstractmethod
    def expects_overwrite(self) -> bool:
        """Whether this Tiler expects refined tiles to be written back to the original image before calling
        get_tile(...) again.

        This is a property of the Tiler, because it typically has implications in the tile masking logic. In other
        words, it would not make sense to overwrite the original image when using a Tiler that has not been designed for
        this.
        """
        ...
