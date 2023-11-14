from abc import ABC, abstractmethod

import torch

from invokeai.backend.tiled_refinement.tile import Tile


class BaseRefiner(ABC):
    @abstractmethod
    def refine(self, image_tile: Tile) -> torch.Tensor:
        """Refine the 'image_tile'.

        Args:
            image_tile (Tile): The tile to refine.

        Returns:
            torch.Tensor: A refined version of 'image_tile.image'. The shape and dtype must match 'image_tile.image'.
        """
        ...
