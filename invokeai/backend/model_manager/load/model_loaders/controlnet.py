# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for ControlNet model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from diffusers import ControlNetModel

import invokeai.backend.assets.sd_base_conf_files as conf_file_cache
from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.config import ControlNetCheckpointConfig, SubModelType
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.ControlNet, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
class ControlNetLoader(GenericDiffusersLoader):
    """Class to load ControlNet models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        config_dirs = {
            BaseModelType.StableDiffusion1: "controlnet_sd15",
            BaseModelType.StableDiffusionXL: "controlnet_sdxl",
        }
        try:
            config_dir = config_dirs[config.base]
        except KeyError:
            raise Exception(f"No configuration template known for controlnet model with base={config.base}")

        if isinstance(config, ControlNetCheckpointConfig):
            return ControlNetModel.from_single_file(
                config.path,
                config=Path(conf_file_cache.__path__[0], config_dir).as_posix(),
                local_files_only=True,
                torch_dtype=self._torch_dtype,
            )
        else:
            return super()._load_model(config, submodel_type)
