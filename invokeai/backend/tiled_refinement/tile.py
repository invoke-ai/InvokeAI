import torch


class XYCoords:
    def __init__(self, x: int, y: int):
        self.x = x  # Row
        self.y = y  # Col


class Tile:
    def __init__(self, image: torch.Tensor, coords: XYCoords, refinement_mask: torch.Tensor, write_mask: torch.Tensor):
        # TODO(ryand): Provide more detail about the image and mask representations in this docstring.
        """Initialize a Tile.

        Args:
            image (torch.Tensor): The tile contents from the original image.
            coords (XYCoords): The coordinates of the tile in the original image.
            refinement_mask (torch.Tensor): A mask to be used during the refinement step.
            write_mask (torch.Tensor): A mask used when writing a refined tile to the output image.
        """
        self.image = image
        self.coords = coords
        self.refinement_mask = refinement_mask
        self.write_mask = write_mask
