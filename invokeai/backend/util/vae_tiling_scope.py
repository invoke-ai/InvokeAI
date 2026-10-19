"""Scoped VAE tiling state, and the tile size a node's field resolves to.

The VAE instances these helpers take belong to the model cache and are shared across invocations and
across *nodes*: the FLUX.1 autoencoder is reached by nine call sites, and the diffusers
`AutoencoderKL` it is loaded as also serves Z-Image. Tiling is a property of one decode, not of the
model, so it has to be restored rather than merely turned off -- `disable_tiling()` clears the flag
but leaves the geometry, and most of the consumers never touch the flag at all.

Two siblings solve the same problem for their own classes: `patch_qwen_image_vae_tiling` for the
Qwen-Image and Wan VAEs, whose geometry is named `tile_sample_min_height`/`tile_sample_stride_*` and
which have no `tile_overlap_factor` at all, and `stable_diffusion.vae_tiling.patch_vae_tiling_params`
for SD's, which leaves `use_tiling` to the caller.

`diffusers_latent_tile` and `diffusers_vae_downsample` state constraints of the diffusers VAE
classes rather than of this module, which is why they are public: the SD sibling derived the tile
geometry independently and produced wrong-sized images for it, and it now calls these instead. A
third copy of that arithmetic is the failure mode this module exists to stop.
"""

from contextlib import contextmanager
from typing import Any, Iterator

# Tile geometry for tiled encode and decode, in output-pixel units. 512px is the geometry the
# diffusers VAEs and the Anima node use.
DEFAULT_TILE_SAMPLE_MIN_SIZE = 512

# A cost floor, not a correctness one: the geometry stays valid all the way down, but the tile count
# grows with the inverse square of the tile size. At 2048x2048 a 128px tile already emits 289 tiles;
# a 16px tile would emit ~16k, and the per-tile kernel-launch overhead dominates long before that.
# Small tiles are also measurably less accurate, because more tiles mean the blend bands sit closer
# to the tiles' own borders, so the low end of a node field is clamped rather than honoured.
MIN_TILE_SAMPLE_SIZE = 128

# Attributes that carry tiling state on the `AutoencoderKL` family. Read defensively -- a class that
# has none of them simply has nothing to restore.
_TILING_ATTRS = (
    "use_tiling",
    "tile_sample_min_size",
    "tile_latent_min_size",
    "tile_overlap_factor",
)

_MISSING = object()

# Floor for the latent tile `diffusers_latent_tile` picks, and the bottom of its search. Below this
# the tile stops being one: a 2-latent tile decodes a band narrower than the blend the overlap
# factor asks for, and a 1-latent one is a single feature column.
_MIN_DIFFUSERS_LATENT_TILE = 4


def resolve_tile_size(tile_size: int) -> int:
    """Resolve a node's ``tile_size`` field to the size the VAE will actually be given.

    ``tile_size <= 0`` is the nodes' "use the default" sentinel -- the workflow UI cannot represent
    ``None`` in a number input and sends 0, and a negative value is not worth failing a generation
    over. It resolves to the module-level default rather than to whatever is currently set on the
    VAE: the instance belongs to the model cache, so reading it back would return whatever the
    previous invocation left there.
    """
    if tile_size <= 0:
        return DEFAULT_TILE_SAMPLE_MIN_SIZE
    return max(tile_size, MIN_TILE_SAMPLE_SIZE)


@contextmanager
def scoped_vae_tiling(vae: Any, tile_size: int | None) -> Iterator[None]:
    """Set the VAE's tiling state for the duration of the block, then restore exactly what was there.

    `tile_size=None` runs in a single pass; `0` is the node fields' sentinel and resolves to
    `DEFAULT_TILE_SAMPLE_MIN_SIZE`, *not* to the VAE's own `sample_size` -- which is 1024 for the
    FLUX.1 autoencoder and would not tile a 1024px image at all. Any other value is the tile size in
    output pixels, floored at `MIN_TILE_SAMPLE_SIZE`.
    """
    original = {name: getattr(vae, name, _MISSING) for name in _TILING_ATTRS}
    try:
        if tile_size is None:
            vae.disable_tiling()
        else:
            # Diffusers' `AutoencoderKL.enable_tiling()` takes no arguments and leaves the geometry
            # at the VAE's own `sample_size` -- 1024 for the FLUX.1 autoencoder, which means a
            # 1024px decode does not tile at all while the caller's working-memory reservation
            # assumes it did. Setting the two attributes afterwards is how the rest of this codebase
            # sizes a diffusers VAE's tiles; see `flux2/ref_image_extension.py`, which forces 512
            # for the same reason. Both are in `_TILING_ATTRS`, so the finally block puts them back.
            downsample = diffusers_vae_downsample(vae)
            latent_tile = diffusers_latent_tile(resolve_tile_size(tile_size), downsample, vae.tile_overlap_factor)
            vae.enable_tiling()
            vae.tile_latent_min_size = latent_tile
            vae.tile_sample_min_size = latent_tile * downsample
        yield
    finally:
        for name, value in original.items():
            if value is not _MISSING:
                setattr(vae, name, value)


def diffusers_vae_downsample(vae: Any) -> int:
    """Output pixels per latent element along one axis, for the diffusers VAE classes.

    `AutoencoderTiny` names the ratio directly and uses that attribute in its own tiling; the
    `AutoencoderKL` family implies it from the number of blocks.
    """
    spatial_scale_factor = getattr(vae, "spatial_scale_factor", None)
    if spatial_scale_factor is not None:
        return int(spatial_scale_factor)
    return 2 ** (len(vae.config.block_out_channels) - 1)


def diffusers_latent_tile(requested_sample_size: int, downsample: int, overlap_factor: float) -> int:
    """The latent tile to give a diffusers VAE, at or below the requested size in output pixels
    unless the request itself is under `_MIN_DIFFUSERS_LATENT_TILE` latents, which the floor wins.

    `tile_sample_min_size` and `tile_latent_min_size` are not two independent knobs. `AutoencoderKL`
    steps its latent loop by `int(tile_latent_min_size * (1 - tile_overlap_factor))` but crops each
    decoded tile to `tile_sample_min_size - int(tile_sample_min_size * tile_overlap_factor)` output
    pixels, and `_tiled_encode` mirrors that -- a pixel step against a latent crop. Those two
    distances describe the same span only for some tiles; for the rest the loop and the crop
    disagree and the assembled result is the wrong size. Measured on a 768x768 decode, 60 of the 84
    legal tile sizes from 128 to 792 came out between 770 and 810 pixels at the stock 0.25 factor,
    the 24 correct ones being exactly the multiples of 32.

    So the latent tile is chosen first -- the largest at or below the request for which the two
    agree -- and the caller derives the sample size from it. The expressions are evaluated as
    diffusers evaluates them rather than reasoned about, because which tiles agree depends on how
    the factor rounds in binary, not on its value as a fraction: at an 8x downsample 0.25 admits
    every multiple of 4, 0.3 only multiples of 10, and 0.35 only multiples of 20.

    Only the decode expression is evaluated. The encode one admits exactly the same tiles -- swept
    over 7 overlap factors x 4 downsample ratios x the first 300 latent tiles, the two agree on
    every single one -- so checking both would be the same question asked twice.
    """
    largest = max(_MIN_DIFFUSERS_LATENT_TILE, requested_sample_size // downsample)
    for latent_tile in range(largest, _MIN_DIFFUSERS_LATENT_TILE - 1, -1):
        sample_size = latent_tile * downsample
        if int(latent_tile * (1 - overlap_factor)) * downsample == sample_size - int(sample_size * overlap_factor):
            return latent_tile
    # No tile in range works: the smallest one this factor admits is above the request, if it
    # admits any at all. Unreachable for the 0.25 every AutoencoderKL ships, which admits every
    # multiple of 4; possible only for a hand-set factor -- 0.35 admits nothing below 20 latents,
    # so an 8x VAE finds nothing for a request of 128 to 152. Honour the request rather than
    # failing the decode over it.
    return largest
