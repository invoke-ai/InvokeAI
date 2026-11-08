"""The cuBLAS block-scale tile layout, shared by every scheme whose scales arrive in it.

cuBLAS and TensorRT take a block-scale operand tiled rather than row-major, and producers write that
tiling into a tensor of the row-major *shape* -- so nothing about the shape gives it away, and a
grid read as stored decodes to noise that loads without complaint. NVIDIA documents it as "Block
Scaling Factors Layout"; ComfyUI writes it with `to_blocked` (`comfy/float.py`).

Two schemes in this tree arrive in it, with different block widths and different scale encodings:
nvfp4 (16-element blocks, ``float8_e4m3fn`` entries) and MXFP8 (32-element blocks, E8M0 exponent
bytes). The tiling is over the *grid*, so it is the same permutation for both and belongs to
neither scheme -- which is why it lives here rather than in one of them importing the other.

Not every producer tiles. NVIDIA's ModelOpt exports, Black Forest Labs' FLUX.3 nvfp4 build among them,
store the grid row-major; :func:`block_scale_grid` tiles such a grid once at load, so that every decode
here reads one layout.
"""

import torch

# A tile spans 128 rows and 4 blocks, whatever a block is worth.
TILE_ROWS = 128
TILE_BLOCKS = 4


def check_tile_layout(rows: int, blocks: int) -> None:
    """Refuse a grid the inverse below cannot address.

    ``to_blocked`` pads a grid that does not divide evenly and the padding is cropped on the way
    back. No published checkpoint measured so far needs it -- every grid in the nvfp4 and MXFP8
    builds checked is already aligned -- so the padded case is refused rather than implemented from
    the specification and shipped untested.
    """
    if rows % TILE_ROWS or blocks % TILE_BLOCKS:
        raise ValueError(
            f"a {rows}x{blocks} block-scale grid is not in the cuBLAS tile layout: rows must be a multiple of "
            f"{TILE_ROWS} and blocks a multiple of {TILE_BLOCKS}. Padded grids are not implemented."
        )


def unblock_scale_grid(scale: torch.Tensor) -> torch.Tensor:
    """Reorder a block-scale grid from cuBLAS's tiled layout into row-major ``[rows, blocks]``.

    The grid is cut into tiles of 128 rows and 4 blocks, the tiles are stored in row-major order,
    and inside a tile the entries run over ``row % 32``, then ``row // 32``, then the block. It is a
    permutation, and inverting it is a reshape: stored ``(tile row, tile column, row % 32,
    row // 32, block)`` becomes ``(tile row, row // 32, row % 32, tile column, block)``. Tile row
    ``r`` therefore occupies stored rows ``[128 r, 128 (r + 1))``.
    """
    if scale.dim() != 2:
        raise ValueError(f"expected a 2-D block-scale grid, got shape {tuple(scale.shape)}")
    rows, blocks = scale.shape
    check_tile_layout(rows, blocks)
    tiles = scale.reshape(rows // TILE_ROWS, blocks // TILE_BLOCKS, 32, 4, TILE_BLOCKS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(rows, blocks)


def block_scale_grid(grid: torch.Tensor) -> torch.Tensor:
    """Lay a row-major ``[rows, blocks]`` grid out in cuBLAS's tiled layout: the inverse of
    :func:`unblock_scale_grid`.

    Row-major ``(tile row, row // 32, row % 32, tile column, block)`` becomes the stored ``(tile row,
    tile column, row % 32, row // 32, block)``. It is the same permutation of axes as the unblocking one,
    which swaps two axes and so undoes itself.
    """
    if grid.dim() != 2:
        raise ValueError(f"expected a 2-D block-scale grid, got shape {tuple(grid.shape)}")
    rows, blocks = grid.shape
    check_tile_layout(rows, blocks)
    tiles = grid.reshape(rows // TILE_ROWS, 4, 32, blocks // TILE_BLOCKS, TILE_BLOCKS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(rows, blocks)
