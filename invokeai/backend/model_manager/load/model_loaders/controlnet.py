# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for ControlNet model loading in InvokeAI."""

from pathlib import Path

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
            and (dest_path / "config.json").stat().st_mtime >= (config.converted_at or 0.0)
            and (dest_path / "config.json").stat().st_mtime >= model_path.stat().st_mtime
        ):
            return False
        else:
            return True

    def _convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Path) -> Path:
        assert isinstance(config, CheckpointConfigBase)
        config_file = config.config_path

        image_size = (
            512
            if config.base == BaseModelType.StableDiffusion1
            else 768
            if config.base == BaseModelType.StableDiffusion2
            else 1024
        )

        self._logger.info(f"Converting {model_path} to diffusers format")
        with open(self._app_config.root_path / config_file, "r") as config_stream:
            convert_controlnet_to_diffusers(
                model_path,
                output_path,
                original_config_file=config_stream,
                image_size=image_size,
                precision=self._torch_dtype,
                from_safetensors=model_path.suffix == ".safetensors",
            )
        return output_path
