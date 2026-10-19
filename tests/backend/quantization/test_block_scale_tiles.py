"""The cuBLAS block-scale tiling, and its inverse.

Every scheme whose scales arrive tiled depends on this one permutation, and reading a tiled grid as
row-major is silent: it pairs blocks with the wrong rows, the checkpoint loads, and the model
generates noise. So the expectation here is built from the index formula measured against real
files, written out explicitly, rather than from the implementation's reshape.
"""

import pytest
import torch

from invokeai.backend.quantization.block_scale_tiles import unblock_scale_grid
from tests.fixtures.quantized_payloads import stored_layout


@pytest.mark.parametrize("shape", [(128, 4), (128, 8), (256, 16), (384, 8)])
def test_a_stored_grid_is_restored_to_row_major(shape: tuple[int, int]) -> None:
    """(384, 8) is three tile rows by two tile columns, so swapping the tile axes cannot pass."""
    grid = torch.arange(shape[0] * shape[1], dtype=torch.float64).reshape(shape)

    assert torch.equal(unblock_scale_grid(stored_layout(grid)), grid)


@pytest.mark.parametrize("shape", [(100, 8), (128, 6)])
def test_a_grid_outside_the_tile_layout_is_refused_rather_than_guessed(shape: tuple[int, int]) -> None:
    """`to_blocked` pads such a grid and the padding is cropped on the way back. No published
    checkpoint needs it, so it is refused rather than implemented from the spec and shipped
    untested."""
    with pytest.raises(ValueError, match="tile layout"):
        unblock_scale_grid(torch.zeros(shape))


def test_a_one_dimensional_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="2-D"):
        unblock_scale_grid(torch.zeros(512))
