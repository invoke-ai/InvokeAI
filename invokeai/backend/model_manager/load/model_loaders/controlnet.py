# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for ControlNet model loading in InvokeAI."""

from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

from invokeai.backend.model_manager import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.config import CheckpointConfigBase
from invokeai.backend.model_manager.convert_ckpt_to_diffusers import convert_controlnet_to_diffusers

from .. import ModelLoaderRegistry
from .generic_diffusers import GenericDiffusersLoader


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.ControlNet, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
class ControlNetLoader(GenericDiffusersLoader):
    """Class to load ControlNet models."""

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        if not isinstance(config, CheckpointConfigBase):
            return False
        elif (
            dest_path.exists()
            and (dest_path / "config.json").stat().st_mtime >= (config.last_modified or 0.0)
            and (dest_path / "config.json").stat().st_mtime >= model_path.stat().st_mtime
        ):
            return False
        else:
            return True

    def _convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Path) -> Path:
        if config.base not in {BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2}:
            raise Exception(f"ControlNet conversion not supported for model type: {config.base}")
        else:
            assert isinstance(config, CheckpointConfigBase)
            config_file = config.config

        if model_path.suffix == ".safetensors":
            checkpoint = safetensors_load_file(model_path, device="cpu")
        else:
            checkpoint = torch.load(model_path, map_location="cpu")

        # sometimes weights are hidden under "state_dict", and sometimes not
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        convert_controlnet_to_diffusers(
            model_path,
            output_path,
            original_config_file=self._app_config.root_path / config_file,
            image_size=512,
            scan_needed=True,
            from_safetensors=model_path.suffix == ".safetensors",
        )
        return output_path
