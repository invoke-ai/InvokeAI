"""The final decode's latent: Ideogram 4 denormalizes per packed channel, then unpatchifies 2x2 patches."""

import torch

from invokeai.backend.ideogram4.sampling_utils import LATENT_DIM, PATCH_SIZE, unpatchify_and_denormalize


def test_unpatchify_and_denormalize_scales_each_packed_channel_then_places_its_patch_pixel() -> None:
    ae_channels = LATENT_DIM // (PATCH_SIZE * PATCH_SIZE)
    packed = torch.randn(1, LATENT_DIM, 3, 2, dtype=torch.float64)
    scale = torch.rand(LATENT_DIM, dtype=torch.float64) + 0.5
    shift = torch.randn(LATENT_DIM, dtype=torch.float64)

    z = unpatchify_and_denormalize(packed, shift, scale)

    assert z.shape == (1, ae_channels, 3 * PATCH_SIZE, 2 * PATCH_SIZE)
    # Packed channel (i * PATCH_SIZE + j) * ae_channels + c holds pixel (i, j) of channel c's patch.
    for c in range(ae_channels):
        for i in range(PATCH_SIZE):
            for j in range(PATCH_SIZE):
                k = (i * PATCH_SIZE + j) * ae_channels + c
                expected = packed[0, k] * scale[k] + shift[k]
                assert torch.equal(z[0, c, i::PATCH_SIZE, j::PATCH_SIZE], expected), (c, i, j)
