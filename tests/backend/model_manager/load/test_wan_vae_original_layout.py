"""Wan VAE single files in the original layout (ComfyUI repacks) load through the Wan registration.

Wan 2.2's 48-channel VAE nests each stage's layers one level deeper than Wan 2.1's
(`decoder.upsamples.<i>.upsamples.<j>`), a layout diffusers' `convert_wan_vae_to_diffusers` does not read.
The expected names below are the ones in `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`'s `wan2.2_vae.safetensors`
and in diffusers' `AutoencoderKLWan` built with the TI2V-5B config.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Wan_Config
from invokeai.backend.model_manager.load.model_loaders.vae import (
    VAELoader,
    _convert_original_wan_vae_to_diffusers,
)

# One key from each rule, Wan 2.2 nested blocks and the names both versions share.
_WAN22_ORIGINAL_LAYOUT = {
    "conv1.weight": "quant_conv.weight",
    "conv2.bias": "post_quant_conv.bias",
    "encoder.conv1.weight": "encoder.conv_in.weight",
    "encoder.downsamples.0.downsamples.0.residual.0.gamma": "encoder.down_blocks.0.resnets.0.norm1.gamma",
    "encoder.downsamples.1.downsamples.0.residual.2.weight": "encoder.down_blocks.1.resnets.0.conv1.weight",
    "encoder.downsamples.1.downsamples.0.shortcut.weight": "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
    "encoder.downsamples.0.downsamples.2.resample.1.weight": "encoder.down_blocks.0.downsampler.resample.1.weight",
    "encoder.downsamples.1.downsamples.2.time_conv.bias": "encoder.down_blocks.1.downsampler.time_conv.bias",
    "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
    "decoder.conv1.weight": "decoder.conv_in.weight",
    "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
    "decoder.upsamples.2.upsamples.1.residual.3.gamma": "decoder.up_blocks.2.resnets.1.norm2.gamma",
    "decoder.upsamples.3.upsamples.2.residual.6.bias": "decoder.up_blocks.3.resnets.2.conv2.bias",
    "decoder.upsamples.0.upsamples.3.time_conv.weight": "decoder.up_blocks.0.upsampler.time_conv.weight",
    "decoder.upsamples.2.upsamples.3.resample.1.bias": "decoder.up_blocks.2.upsampler.resample.1.bias",
    "decoder.head.2.weight": "decoder.conv_out.weight",
}


def test_wan22_original_layout_converts_to_diffusers_names() -> None:
    converted = _convert_original_wan_vae_to_diffusers({key: torch.zeros(1) for key in _WAN22_ORIGINAL_LAYOUT})

    assert set(converted) == set(_WAN22_ORIGINAL_LAYOUT.values())


def test_wan21_original_layout_still_converts_like_diffusers() -> None:
    """Wan 2.1's flat stages (`decoder.upsamples.<i>.residual`) are left to diffusers' converter."""
    converted = _convert_original_wan_vae_to_diffusers(
        {
            "decoder.upsamples.4.residual.0.gamma": torch.zeros(1),
            "encoder.downsamples.2.resample.1.weight": torch.zeros(1),
        }
    )

    assert set(converted) == {"decoder.up_blocks.1.resnets.0.norm1.gamma", "encoder.down_blocks.2.resample.1.weight"}


def test_wan_registration_converts_an_original_layout_file(tmp_path: Path) -> None:
    """The Wan registration loaded the state dict as-is, so a ComfyUI-layout file could not fill the module."""
    path = tmp_path / "wan2.2_vae.pth"
    torch.save({key: torch.zeros(1) for key in _WAN22_ORIGINAL_LAYOUT}, path)
    loader = VAELoader.__new__(VAELoader)
    loader._torch_dtype = torch.float32  # type: ignore[attr-defined]
    loader._ram_cache = MagicMock()  # type: ignore[attr-defined]

    with (
        patch("diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan") as wan,
        patch("invokeai.backend.model_manager.load.model_loaders.vae.load_state_dict_ignoring_extras") as load,
        patch("invokeai.backend.wan.rocm_causal_conv3d.patch_wan_causal_conv3d_for_rocm"),
    ):
        loader._load_model(VAE_Checkpoint_Wan_Config.model_construct(path=str(path), latent_channels=48))

    assert wan.call_args.kwargs["z_dim"] == 48
    (_, loaded), _ = load.call_args
    assert set(loaded) == set(_WAN22_ORIGINAL_LAYOUT.values())
