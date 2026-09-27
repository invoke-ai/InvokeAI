"""Shared fixtures for the VAE invocation tests."""

from collections.abc import Callable

import pytest
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


def _build_flux_shaped_vae() -> AutoencoderKL:
    """A real `AutoencoderKL` with the FLUX.1 VAE's *shape* and almost none of its weight.

    8059 parameters against 84 million, built in ~6 ms, and it decodes an 8x8 latent in ~7 ms —
    but it is the genuine class, so the properties these tests are about are the real ones rather
    than a mock's:

    * four blocks, hence the same 8x spatial compression, which is what `diffusers_latent_tile`
      computes its geometry from;
    * `tile_overlap_factor`, `tile_sample_min_size`, `tile_latent_min_size` and `use_tiling` as
      instance attributes the tiling scope actually reads and restores;
    * `latent_channels`, `scaling_factor` and `shift_factor`, which is what `is_flux_family_vae`
      reads -- the class alone no longer says which latent space a VAE encodes into;
    * a working `tiled_decode`/`tiled_encode`, so a cell can assert the assembled shape rather
      than only that a flag was set.

    A `MagicMock(spec=AutoencoderKL)` cannot stand in for it: `tile_overlap_factor` is set in
    `__init__`, so it is not on the class and a spec'd mock does not have it at all — the tiling
    scope would raise `AttributeError` on a double that looked correct.
    """
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=2,
        sample_size=64,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )


@pytest.fixture
def flux_shaped_vae() -> Callable[[], AutoencoderKL]:
    """Factory, not an instance: a test that needs two independent VAEs should not share one."""
    return _build_flux_shaped_vae
