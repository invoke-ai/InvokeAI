"""Tile geometry for the SD1.5/SDXL VAE nodes.

`patch_vae_tiling_params` used to take `tile_sample_min_size` and `tile_latent_min_size` as two free
parameters, and both call sites derived the latter as `tile_size // 8`. Those two attributes are not
independent on `AutoencoderKL`: its tiled decode steps the latent loop by one of them and crops each
decoded tile by the other, and its tiled encode mirrors that. Deriving them separately produced a
wrong-sized image for every tile size the two disagree on -- which, at the stock 0.25 overlap
factor, is every value the `multiple_of=8` node field accepts that is not also a multiple of 32.
"""

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.backend.stable_diffusion.vae_tiling import patch_vae_tiling_params

# What both nodes pass, and what the stock AutoencoderKL uses.
OVERLAP_FACTOR = 0.25

# Every value the `tile_size` field accepts between its practical floor and a 512px canvas. The
# node treats 0 as "leave the VAE's own geometry alone" and never enters this path.
FIELD_VALUES = range(128, 520, 8)


@pytest.fixture(scope="module")
def vae() -> AutoencoderKL:
    """A structurally faithful SD VAE: four blocks, so the real 8x latent grid, with the channel
    counts cut to what keeps ~100 decodes cheap on CPU."""
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(4, 8, 16, 16),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1,
        norm_num_groups=4,
        sample_size=512,
    ).eval()


@pytest.fixture(scope="module")
def tiny_vae() -> AutoencoderTiny:
    return AutoencoderTiny(
        encoder_block_out_channels=(4, 4, 4, 4),
        decoder_block_out_channels=(4, 4, 4, 4),
    ).eval()


class TestEveryFieldValueProducesTheRightShape:
    def test_tiled_decode_keeps_the_full_image_size(self, vae):
        """Before the derivation was shared, 36 of these 49 values assembled an image between 518
        and 542 pixels wide instead of 512 -- silently, and on the most-used decode path."""
        vae.enable_tiling()
        latents = torch.zeros(1, 4, 64, 64)

        wrong: list[tuple[int, int]] = []
        for tile_size in FIELD_VALUES:
            with torch.no_grad(), patch_vae_tiling_params(vae, tile_size, OVERLAP_FACTOR):
                decoded = vae.decode(latents, return_dict=False)[0]
            if decoded.shape != (1, 3, 512, 512):
                wrong.append((tile_size, decoded.shape[-1]))

        assert not wrong

    def test_tiled_encode_keeps_the_full_latent_size(self, vae):
        """The encode path is affected too, and not symmetrically: `_tiled_encode` steps a *pixel*
        loop and crops a *latent* tile, so the same inconsistency shows up as a latent one or two
        elements too large -- which then propagates into denoising -- or as a conv size error.
        Measured before the fix: 14 of 49 values wrong, plus one RuntimeError at 136.
        """
        vae.enable_tiling()
        image = torch.zeros(1, 3, 512, 512)

        wrong: list[tuple[int, int]] = []
        for tile_size in FIELD_VALUES:
            with torch.no_grad(), patch_vae_tiling_params(vae, tile_size, OVERLAP_FACTOR):
                latents = vae.encode(image).latent_dist.mode()
            if latents.shape != (1, 4, 64, 64):
                wrong.append((tile_size, latents.shape[-1]))

        assert not wrong

    @pytest.mark.parametrize("requested,applied", [(128, 128), (136, 128), (160, 160), (200, 192), (512, 512)])
    def test_the_applied_tile_is_the_largest_usable_one_at_or_below_the_request(self, vae, requested, applied):
        """The user-visible part of the fix: a request the VAE cannot assemble is rounded *down* to
        one it can, never up and never honoured literally. At the stock factor that is the nearest
        multiple of 32, so the tile shrinks by at most 24 output pixels."""
        with patch_vae_tiling_params(vae, requested, OVERLAP_FACTOR):
            assert vae.tile_sample_min_size == applied
            assert vae.tile_latent_min_size == applied // 8


class TestTheTinyVaeIsUnaffected:
    """`AutoencoderTiny` reads only `tile_sample_min_size` in its tiled encode and only
    `tile_latent_min_size` in its tiled decode, so the two never meet and it had no bug to fix. It
    goes through the same derivation rather than a class branch; this pins that that is harmless."""

    def test_tiled_decode_and_encode_keep_their_shapes(self, tiny_vae):
        tiny_vae.enable_tiling()
        for tile_size in FIELD_VALUES:
            with torch.no_grad(), patch_vae_tiling_params(tiny_vae, tile_size, OVERLAP_FACTOR):
                decoded = tiny_vae.decode(torch.zeros(1, 4, 64, 64), return_dict=False)[0]
                encoded = tiny_vae.encode(torch.zeros(1, 3, 512, 512), return_dict=False)[0]
            assert decoded.shape == (1, 3, 512, 512)
            assert encoded.shape == (1, 4, 64, 64)

    def test_its_own_downsample_ratio_is_used_rather_than_a_constant(self, tiny_vae):
        """It names the ratio `spatial_scale_factor` instead of implying it from the block count,
        and that is the attribute its own tiling reads."""
        with patch_vae_tiling_params(tiny_vae, 256, OVERLAP_FACTOR):
            assert tiny_vae.tile_sample_min_size == tiny_vae.tile_latent_min_size * tiny_vae.spatial_scale_factor


class TestTheVaesOwnGeometryIsRestored:
    """The VAE belongs to the model cache and outlives the invocation, so a tile size set for one
    decode must not become the default for the next."""

    @pytest.mark.parametrize("factory", ["vae", "tiny_vae"])
    def test_all_three_attributes_come_back(self, request, factory):
        subject = request.getfixturevalue(factory)
        before = (subject.tile_sample_min_size, subject.tile_latent_min_size, subject.tile_overlap_factor)

        with patch_vae_tiling_params(subject, 256, 0.5):
            assert subject.tile_overlap_factor == 0.5

        assert (subject.tile_sample_min_size, subject.tile_latent_min_size, subject.tile_overlap_factor) == before

    def test_an_exception_still_restores(self, vae):
        before = (vae.tile_sample_min_size, vae.tile_latent_min_size, vae.tile_overlap_factor)

        with pytest.raises(RuntimeError, match="boom"):
            with patch_vae_tiling_params(vae, 256, OVERLAP_FACTOR):
                raise RuntimeError("boom")

        assert (vae.tile_sample_min_size, vae.tile_latent_min_size, vae.tile_overlap_factor) == before
