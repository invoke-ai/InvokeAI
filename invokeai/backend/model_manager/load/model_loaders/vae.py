# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

from typing import Optional

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Anima_Config,
    VAE_Checkpoint_Config_Base,
    VAE_Checkpoint_QwenImage_Config,
    VAE_Checkpoint_Wan_Config,
    VAE_Diffusers_Wan_Config,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


# Architectural defaults for the Wan 2.2-VAE (TI2V-5B). Verbatim from the
# vae/config.json shipped with Wan-AI/Wan2.2-TI2V-5B-Diffusers — only the
# values that differ from diffusers' AutoencoderKLWan defaults are listed.
# latents_mean / latents_std are required because the model normalises latents
# against them at encode/decode time; the wrong arrays produce silent garbage.
_WAN_TI2V_5B_VAE_CONFIG: dict = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "is_residual": True,
    "latents_mean": [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ],
    "latents_std": [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ],
}


def _wan_vae_init_kwargs_for(latent_channels: int) -> dict:
    """Return the AutoencoderKLWan constructor kwargs for a given z_dim.

    z_dim=48 means TI2V-5B's Wan 2.2-VAE (different base dim, patchified IO,
    16x spatial). Anything else falls back to the A14B / Wan 2.1 defaults.
    """
    if latent_channels == 48:
        return dict(_WAN_TI2V_5B_VAE_CONFIG)
    return {"z_dim": latent_channels}


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class VAELoader(GenericDiffusersLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, VAE_Checkpoint_Anima_Config):
            from diffusers.models.autoencoders import AutoencoderKLWan

            return AutoencoderKLWan.from_single_file(
                config.path,
                torch_dtype=self._torch_dtype,
            )
        elif isinstance(config, VAE_Checkpoint_Wan_Config):
            return self._load_wan_vae(config)
        elif isinstance(config, VAE_Diffusers_Wan_Config):
            return self._load_wan_vae_diffusers(config)
        elif isinstance(config, VAE_Checkpoint_QwenImage_Config):
            return self._load_qwen_image_vae(config)
        elif isinstance(config, VAE_Checkpoint_Config_Base):
            return AutoencoderKL.from_single_file(
                config.path,
                torch_dtype=self._torch_dtype,
            )
        else:
            return super()._load_model(config, submodel_type)

    def _load_wan_vae(self, config: VAE_Checkpoint_Wan_Config) -> AnyModel:
        """Load a Wan 2.2 VAE from a single safetensors file.

        Picks the correct ``AutoencoderKLWan`` config based on ``z_dim``. The Wan
        ecosystem ships two distinct VAE architectures:

        * ``z_dim=16`` — the Wan 2.1 / Wan 2.2 A14B VAE. Diffusers' defaults match
          this one (base_dim=96, 8x spatial, no patchify, 3 in/out channels).
        * ``z_dim=48`` — the Wan 2.2-VAE used by TI2V-5B. Larger (base_dim=160,
          decoder_base_dim=256), 16x spatial, patchify with patch_size=2 (so
          in/out channels are 12 = 3 RGB x 2x2 patch), residual blocks, and
          its own latents_mean / latents_std.

        Without overriding those params at construction time, the state dict
        from the TI2V-5B VAE checkpoint won't load (channel and shape mismatches
        throughout the encoder + decoder).
        """
        import accelerate
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        from safetensors.torch import load_file

        sd = load_file(config.path)

        if self._torch_dtype is not None:
            for k in list(sd.keys()):
                if sd[k].is_floating_point():
                    sd[k] = sd[k].to(self._torch_dtype)

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        init_kwargs = _wan_vae_init_kwargs_for(config.latent_channels)
        with accelerate.init_empty_weights():
            model = AutoencoderKLWan(**init_kwargs)

        model.load_state_dict(sd, strict=True, assign=True)
        model.eval()
        return model

    def _load_wan_vae_diffusers(self, config: VAE_Diffusers_Wan_Config) -> AnyModel:
        """Load a Wan 2.2 VAE from a flat diffusers folder (AutoencoderKLWan).

        The standalone install ``Wan-AI/Wan2.2-T2V-A14B-Diffusers::vae`` lands as a
        single-class folder (``config.json`` + ``diffusion_pytorch_model.safetensors``,
        no ``model_index.json``). The generic loader rejects this when a
        ``submodel_type`` is requested — we always pass ``SubModelType.VAE`` from
        the model loader invocation since that's how cached entries are keyed.
        Loading ``AutoencoderKLWan`` directly here sidesteps the submodel check.

        Forces bfloat16 (same as ``WanDiffusersModel``) — fp16 is unstable on the
        Wan VAE.
        """
        import torch
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

        return AutoencoderKLWan.from_pretrained(
            config.path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    def _load_qwen_image_vae(self, config: VAE_Checkpoint_QwenImage_Config) -> AnyModel:
        """Load a Qwen Image VAE from a single safetensors file.

        The Qwen Image VAE checkpoint is expected to be in the diffusers state-dict
        layout (i.e. the same keys as `vae/diffusion_pytorch_model.safetensors` from
        the Qwen-Image repo). `AutoencoderKLQwenImage` does not register a single-file
        conversion in diffusers, so we instantiate the model with default config and
        load the state dict directly.
        """
        import accelerate
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
        from safetensors.torch import load_file

        sd = load_file(config.path)

        if self._torch_dtype is not None:
            for k in list(sd.keys()):
                if sd[k].is_floating_point():
                    sd[k] = sd[k].to(self._torch_dtype)

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        with accelerate.init_empty_weights():
            model = AutoencoderKLQwenImage()

        model.load_state_dict(sd, strict=True, assign=True)
        model.eval()
        return model
