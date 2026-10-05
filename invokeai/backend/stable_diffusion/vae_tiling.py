from contextlib import contextmanager

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.backend.util.vae_tiling_scope import diffusers_latent_tile, diffusers_vae_downsample


@contextmanager
def patch_vae_tiling_params(
    vae: AutoencoderKL | AutoencoderTiny,
    tile_sample_min_size: int,
    tile_overlap_factor: float,
):
    """Patch the parameters that control the VAE tiling tile size and overlap.

    These parameters are not explicitly exposed in the VAE's API, but they have a significant impact on the quality of
    the outputs. As a general rule, bigger tiles produce better results, but this comes at the cost of higher memory
    usage.

    `tile_sample_min_size` is a request, in output pixels. The tile actually applied is the largest one at or below it
    that the VAE's tiled paths can assemble -- see `diffusers_latent_tile` for why that is not every value, and what
    setting `tile_latent_min_size` independently used to cost. That derivation is the reason this takes a single size
    rather than the pair it patches: the pair has a constraint between its members, so a caller must not be able to
    supply one.

    `AutoencoderTiny` has no such constraint -- its tiled encode reads only `tile_sample_min_size` and its tiled decode
    only `tile_latent_min_size`, so the two never meet -- but it takes the same derivation rather than a class branch.
    It therefore gets a tile the constraint would have required: measured across every `multiple_of=8` request from 64
    to 4088, up to 3 latent elements (24 output pixels) smaller than asked, worst case at 88px. Shape stays exact
    either way, so the cost is tile size, not correctness -- and a class branch to recover 24px is not worth it.
    """
    # Record initial config.
    orig_tile_sample_min_size = vae.tile_sample_min_size
    orig_tile_latent_min_size = vae.tile_latent_min_size
    orig_tile_overlap_factor = vae.tile_overlap_factor

    downsample = diffusers_vae_downsample(vae)
    latent_tile = diffusers_latent_tile(tile_sample_min_size, downsample, tile_overlap_factor)

    try:
        # Apply target config.
        vae.tile_sample_min_size = latent_tile * downsample
        vae.tile_latent_min_size = latent_tile
        vae.tile_overlap_factor = tile_overlap_factor
        yield
    finally:
        # Restore initial config.
        vae.tile_sample_min_size = orig_tile_sample_min_size
        vae.tile_latent_min_size = orig_tile_latent_min_size
        vae.tile_overlap_factor = orig_tile_overlap_factor
