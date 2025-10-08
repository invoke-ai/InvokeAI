# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for ControlNet model loading in InvokeAI."""

from typing import Optional

from diffusers import ControlNetModel

from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion1, type=ModelType.ControlNet, format=ModelFormat.Diffusers
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion1, type=ModelType.ControlNet, format=ModelFormat.Checkpoint
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion2, type=ModelType.ControlNet, format=ModelFormat.Diffusers
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusion2, type=ModelType.ControlNet, format=ModelFormat.Checkpoint
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusionXL, type=ModelType.ControlNet, format=ModelFormat.Diffusers
)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusionXL, type=ModelType.ControlNet, format=ModelFormat.Checkpoint
)
class ControlNetLoader(GenericDiffusersLoader):
    """Class to load ControlNet models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, ControlNet_Checkpoint_Config_Base):
            return ControlNetModel.from_single_file(
                config.path,
                torch_dtype=self._torch_dtype,
            )
        else:
            return super()._load_model(config, submodel_type)
