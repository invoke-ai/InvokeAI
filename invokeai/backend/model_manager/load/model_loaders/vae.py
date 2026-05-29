# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

from typing import Optional

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Anima_Config,
    VAE_Checkpoint_Config_Base,
    VAE_Checkpoint_QwenImage_Config,
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
        elif isinstance(config, VAE_Checkpoint_QwenImage_Config):
            return self._load_qwen_image_vae(config)
        elif isinstance(config, VAE_Checkpoint_Config_Base):
            return AutoencoderKL.from_single_file(
                config.path,
                torch_dtype=self._torch_dtype,
            )
        else:
            return super()._load_model(config, submodel_type)

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
