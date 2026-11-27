"""The SD3 autoencoder, for where InvokeAI has to know it without a `config.json` beside the weights."""

from collections.abc import Mapping
from typing import Any

# From the `vae/config.json` Stability publishes with SD 3.5 medium and large, which are identical.
# The network is the FLUX.1 autoencoder's -- same 244 tensors, same shapes -- and these two constants,
# which normalise its latents, are the only thing that tells the two apart.
SD3_VAE_SCALING_FACTOR = 1.5305
SD3_VAE_SHIFT_FACTOR = 0.0609


def get_sd3_vae_diffusers_config() -> dict[str, Any]:
    """The SD3 autoencoder as `AutoencoderKL.__init__` keyword arguments.

    For a single-file VAE, which carries weights and no config. Fields left to diffusers' defaults
    (`act_fn`, `norm_num_groups`, `force_upcast`, `mid_block_add_attention`, `latents_mean`,
    `latents_std`) match the published config; `tests/backend/model_manager/load/test_16_channel_vae_loader.py`
    pins the constructed config against it, so a changed default breaks a test rather than the model.
    """
    return {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (128, 256, 512, 512),
        "layers_per_block": 2,
        "latent_channels": 16,
        "sample_size": 1024,
        "scaling_factor": SD3_VAE_SCALING_FACTOR,
        "shift_factor": SD3_VAE_SHIFT_FACTOR,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }


def is_sd3_vae_config(config: Mapping[str, Any]) -> bool:
    """Whether an autoencoder config describes SD3's latent space: 16 channels, normalised with SD3's constants."""
    return (
        config.get("latent_channels") == 16
        and config.get("scaling_factor") == SD3_VAE_SCALING_FACTOR
        and config.get("shift_factor") == SD3_VAE_SHIFT_FACTOR
    )
