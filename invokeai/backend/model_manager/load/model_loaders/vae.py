# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

from typing import Optional

from diffusers import AutoencoderKL

from invokeai.backend.model_manager import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.config import AnyModel, SubModelType, VAECheckpointConfig

from .. import ModelLoaderRegistry
from .generic_diffusers import GenericDiffusersLoader


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class VAELoader(GenericDiffusersLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, VAECheckpointConfig):
            return AutoencoderKL.from_single_file(
                config.path,
                config=self._app_config.legacy_conf_path / config.config_path,
                torch_dtype=self._torch_dtype,
                local_files_only=True,
            )
        else:
            return super()._load_model(config, submodel_type)
