# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as safetensors_load_file

from invokeai.backend.model_manager import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.config import CheckpointConfigBase
from invokeai.backend.model_manager.convert_ckpt_to_diffusers import convert_ldm_vae_to_diffusers

from .. import ModelLoaderRegistry
from .generic_diffusers import GenericDiffusersLoader


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion1, type=ModelType.VAE, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion2, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class VAELoader(GenericDiffusersLoader):
    """Class to load VAE models."""

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        if not isinstance(config, CheckpointConfigBase):
            return False
        elif (
            dest_path.exists()
            and (dest_path / "config.json").stat().st_mtime >= (config.converted_at or 0.0)
            and (dest_path / "config.json").stat().st_mtime >= model_path.stat().st_mtime
        ):
            return False
        else:
            return True

    def _convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Path) -> Path:
        # TODO(MM2): check whether sdxl VAE models convert.
        if config.base not in {BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2}:
            raise Exception(f"VAE conversion not supported for model type: {config.base}")
        else:
            assert isinstance(config, CheckpointConfigBase)
            config_file = config.config_path

        if model_path.suffix == ".safetensors":
            checkpoint = safetensors_load_file(model_path, device="cpu")
        else:
            checkpoint = torch.load(model_path, map_location="cpu")

        # sometimes weights are hidden under "state_dict", and sometimes not
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        ckpt_config = OmegaConf.load(self._app_config.root_path / config_file)
        assert isinstance(ckpt_config, DictConfig)

        vae_model = convert_ldm_vae_to_diffusers(
            checkpoint=checkpoint,
            vae_config=ckpt_config,
            image_size=512,
        )
        vae_model.to(self._torch_dtype)  # set precision appropriately
        vae_model.save_pretrained(output_path, safe_serialization=True)
        return output_path
