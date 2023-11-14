from abc import ABC, abstractmethod

import torch


class BaseTiler(ABC):
    """A base class to be implemented by all tilers.

    Usage:
    ```
    tiler = ConcreteTiler(...)

    tiler.initialize(image)

    for i in range(tiler.get_num_tiles()):
        tile = tiler.read_tile(i)
        # Refine tile ...
        tiler.write_tile(refined_tile, i)

    result = tiler.get_output()
    ```
    """

    @abstractmethod
    def initialize(self, image: torch.Tensor):
        """Perform any pre-processing necessary to initialize the tiler. For example, some tilers will use this method
        to pre-compute the tile coordinates.
        """
        ...

    @abstractmethod
    def get_num_tiles(self) -> int:
        """The number of tiles that will be produced by this Tiler for 'image'."""
        ...

    @abstractmethod
    def read_tile(self, i: int) -> torch.Tensor:
        """Return the 'i'th tile from 'image'."""
        ...

    @abstractmethod
    def write_tile(self, tile_image: torch.Tensor, i: int):
        """Write the refined i'th tile to the output image."""
        ...

    @abstractmethod
    def get_output(self) -> torch.Tensor:
        """Return the tiled output image."""
        ...


class TilerNotInitializedError(Exception):
    """An exception thrown when a Tiler method is called before calling initialize(...)."""

    pass
