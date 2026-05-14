"""Tests for the Wan VAE single-file loader helper.

Covers the bug where ``AutoencoderKLWan`` was always instantiated with the A14B
defaults (base_dim=96, out_channels=3, no patchify), causing the TI2V-5B VAE
checkpoint to fail state_dict loading with shape mismatches throughout the
encoder + decoder. The fix routes z_dim=48 to the TI2V-5B-specific
constructor kwargs.
"""

import accelerate
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from invokeai.backend.model_manager.load.model_loaders.vae import _wan_vae_init_kwargs_for


def test_a14b_returns_default_z_dim_only() -> None:
    # The A14B path should still be the trivial case — only z_dim is overridden,
    # leaving diffusers' defaults (base_dim=96, out_channels=3, etc.) intact.
    assert _wan_vae_init_kwargs_for(16) == {"z_dim": 16}


def test_ti2v_5b_returns_full_architectural_override() -> None:
    kw = _wan_vae_init_kwargs_for(48)
    assert kw["z_dim"] == 48
    assert kw["base_dim"] == 160
    assert kw["decoder_base_dim"] == 256
    assert kw["in_channels"] == 12
    assert kw["out_channels"] == 12
    assert kw["patch_size"] == 2
    assert kw["scale_factor_spatial"] == 16
    assert kw["is_residual"] is True
    # latents_mean/std need to be 48-vectors so the model can construct.
    assert len(kw["latents_mean"]) == 48
    assert len(kw["latents_std"]) == 48


def test_ti2v_5b_kwargs_instantiate_with_expected_shapes() -> None:
    # End-to-end check: the kwargs let AutoencoderKLWan build cleanly and the
    # resulting model carries the TI2V-5B-shaped layers (z_dim=48, decoder
    # outputs 12 channels — this is what failed before the fix).
    with accelerate.init_empty_weights():
        model = AutoencoderKLWan(**_wan_vae_init_kwargs_for(48))
    assert model.z_dim == 48
    assert model.config.base_dim == 160
    assert model.config.decoder_base_dim == 256
    assert model.config.out_channels == 12
    assert model.config.patch_size == 2
    # decoder.conv_out emits the patchified 12-channel output (3 RGB x 2x2 patch).
    assert model.decoder.conv_out.weight.shape[0] == 12


def test_a14b_kwargs_instantiate_with_expected_shapes() -> None:
    with accelerate.init_empty_weights():
        model = AutoencoderKLWan(**_wan_vae_init_kwargs_for(16))
    assert model.z_dim == 16
    assert model.config.base_dim == 96
    assert model.config.out_channels == 3
    assert model.config.patch_size is None
