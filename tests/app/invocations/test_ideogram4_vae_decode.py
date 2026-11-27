"""Ideogram 4 decodes with whichever shape its 32-channel VAE arrives in.

A diffusers Ideogram 4 pipeline bundles the autoencoder in the vendored BFL layout; installing the
same weights on their own (the file Comfy-Org ships beside the single-file transformers) produces a
diffusers `AutoencoderKLFlux2`. The node has to accept both, and they have to decode to the same
image -- otherwise the single-file path would quietly render differently from the bundled one.
"""

import pytest
import torch
from diffusers import AutoencoderKLFlux2

from invokeai.app.invocations.vae.ideogram4_latents_to_image import _decode
from invokeai.backend.ideogram4.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict

# A miniature of the released geometry: four stages and 32 latent channels as shipped, 32 base
# channels instead of 128. `norm_num_groups` stays at the released 32 because the vendored decoder
# hard-codes it -- lowering it here would compare two different normalizations.
TINY_KWARGS = {
    "block_out_channels": (32, 64, 64, 64),
    "latent_channels": 32,
    "layers_per_block": 1,
    "norm_num_groups": 32,
}
TINY_PARAMS = AutoEncoderParams(ch=32, ch_mult=[1, 2, 2, 2], num_res_blocks=1, z_channels=32)


@pytest.fixture(scope="module")
def vaes() -> tuple[AutoencoderKLFlux2, AutoEncoder]:
    torch.manual_seed(0)
    diffusers_vae = AutoencoderKLFlux2(**TINY_KWARGS).eval()
    vendored = AutoEncoder(TINY_PARAMS).eval()
    # Strict: the converter is what makes "the same weights" true, and a key it silently dropped
    # would leave a random tensor in the comparison below.
    vendored.load_state_dict(convert_diffusers_state_dict(diffusers_vae.state_dict()))
    return diffusers_vae, vendored


def test_the_two_layouts_decode_the_same_latent_to_the_same_image(vaes) -> None:
    diffusers_vae, vendored = vaes
    torch.manual_seed(1)
    z = torch.randn(1, 32, 4, 4)

    with torch.no_grad():
        from_vendored = _decode(vendored, z)
        from_diffusers = _decode(diffusers_vae, z)

    assert from_vendored.shape == from_diffusers.shape
    # Float32 rounding, not a tolerance: the vendored decoder runs `post_quant_conv` itself, which
    # is exactly what `AutoencoderKLFlux2.decode` applies before its own decoder.
    assert torch.allclose(from_vendored, from_diffusers, atol=1e-4)


def test_a_vae_from_another_family_is_named_in_the_error(vaes) -> None:
    # Reached only if the loader node's compatibility check is bypassed (a hand-built graph), so it
    # has to say what is wrong rather than raise an AttributeError from inside the decode.
    with pytest.raises(ValueError, match="32-channel"):
        _decode(torch.nn.Linear(1, 1), torch.zeros(1, 32, 4, 4))
