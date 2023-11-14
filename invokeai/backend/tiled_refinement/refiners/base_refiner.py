from abc import ABC, abstractmethod

import torch


class BaseRefiner(ABC):
    @abstractmethod
    def refine(self, image_tile: torch.Tensor) -> torch.Tensor:
        """Refine the 'image_tile'.

        Args:
            image_tile (torch.Tensor): The tile to refine.

        Returns:
            torch.Tensor: A refined version of 'image_tile'. The shape and dtype must match.
        """
        ...
