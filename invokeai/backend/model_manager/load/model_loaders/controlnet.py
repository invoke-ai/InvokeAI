# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for ControlNet model loading in InvokeAI."""

from pathlib import Path

import safetensors
import torch

from invokeai.backend.model_manager import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.convert_ckpt_to_diffusers import convert_controlnet_to_diffusers
from invokeai.backend.model_manager.load.load_base import AnyModelLoader

from .generic_diffusers import GenericDiffusersLoader


@AnyModelLoader.register(base=BaseModelType.Any, type=ModelType.ControlNet, format=ModelFormat.Diffusers)
@AnyModelLoader.register(base=BaseModelType.Any, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
class ControlnetLoader(GenericDiffusersLoader):
    """Class to load ControlNet models."""

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        if config.format != ModelFormat.Checkpoint:
            return False
        elif (
            dest_path.exists()
            and (dest_path / "config.json").stat().st_mtime >= (config.last_modified or 0.0)
            and (dest_path / "config.json").stat().st_mtime >= model_path.stat().st_mtime
        ):
            return False
        else:
            return True

    def _convert_model(self, config: AnyModelConfig, weights_path: Path, output_path: Path) -> Path:
        if config.base not in {BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2}:
            raise Exception(f"Vae conversion not supported for model type: {config.base}")
        else:
            assert hasattr(config, "config")
            config_file = config.config

        if weights_path.suffix == ".safetensors":
            checkpoint = safetensors.torch.load_file(weights_path, device="cpu")
        else:
            checkpoint = torch.load(weights_path, map_location="cpu")

        # sometimes weights are hidden under "state_dict", and sometimes not
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        convert_controlnet_to_diffusers(
            weights_path,
            output_path,
            original_config_file=self._app_config.root_path / config_file,
            image_size=512,
            scan_needed=True,
            from_safetensors=weights_path.suffix == ".safetensors",
        )
        return output_path
