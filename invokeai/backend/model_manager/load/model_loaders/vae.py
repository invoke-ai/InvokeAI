# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

import json
from pathlib import Path
from typing import Optional

import accelerate
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from safetensors import safe_open

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
from invokeai.backend.quantization.sdnq.loaders import raise_on_incomplete_sdnq_load, sdnq_sd_loader


def _is_sdnq_vae_folder(path: Path) -> bool:
    """Check if a VAE folder contains SDNQ-quantized weights.

    Handles arbitrarily named and sharded safetensors directories, matching what the shared
    sdnq_sd_loader() can actually load: a VAE using standard shard files such as
    ``diffusion_pytorch_model-00001-of-00002.safetensors`` must still be detected.
    """
    # Strongest signal: the SDNQ marker file (also covers sharded exports).
    quant_config_path = path / "quantization_config.json"
    if quant_config_path.exists():
        try:
            with open(quant_config_path, "r", encoding="utf-8") as f:
                if json.load(f).get("quant_method") == "sdnq":
                    return True
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: union the keys across every safetensors shard and look for a weight + scale pair. A
    # weight and its scale can live in different shards, so we must consider all shards together.
    all_keys: set[str] = set()
    for shard in sorted(path.glob("*.safetensors")):
        try:
            with safe_open(shard, framework="pt", device="cpu") as f:
                all_keys.update(f.keys())
        except Exception:
            continue
    return any(key.endswith(".weight") and f"{key[:-7]}.scale" in all_keys for key in all_keys)


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

        model_path = Path(config.path)

        # Check if this is an SDNQ-quantized VAE folder
        if model_path.is_dir() and _is_sdnq_vae_folder(model_path):
            return self._load_sdnq_vae(model_path)

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

    def _load_sdnq_vae(self, model_path: Path) -> AnyModel:
        """Load SDNQ-quantized VAE with on-the-fly dequantization."""
        # Find the safetensors source. Prefer a single canonical file; otherwise hand the whole
        # directory to sdnq_sd_loader, which merges arbitrarily named / sharded safetensors files.
        model_file = model_path / "diffusion_pytorch_model.safetensors"
        if not model_file.exists():
            model_file = model_path / "model.safetensors"
        source = model_file if model_file.exists() else model_path

        # Load SDNQ state dict
        sd = sdnq_sd_loader(source, compute_dtype=self._torch_dtype)

        # Create empty model from config
        with accelerate.init_empty_weights():
            model = AutoencoderKL.from_config(AutoencoderKL.load_config(model_path, local_files_only=True))

        # Load state dict with SDNQTensor objects. AutoencoderKL has no tied weights, so a complete
        # state dict is expected — a missing key would leave a required parameter on the meta device.
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        raise_on_incomplete_sdnq_load("SDNQ VAE", missing, unexpected)
        return model
