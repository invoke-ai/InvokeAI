# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Class for Qwen-Image model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from diffusers import DiffusionPipeline

from invokeai.backend.model_manager.config import AnyModelConfig, MainDiffusersConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class QwenImageLoader(ModelLoader):
    """Class to load Qwen-Image models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MainDiffusersConfig):
            raise ValueError("Only MainDiffusersConfig models are currently supported here.")
        
        if config.base != BaseModelType.QwenImage:
            raise ValueError("This loader only supports Qwen-Image models.")
        
        model_path = Path(config.path)
        
        if submodel_type is not None:
            # Load individual submodel components
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=self._torch_dtype,
                variant=config.repo_variant.value if config.repo_variant else None,
            )
            
            # Return the specific submodel
            if hasattr(pipeline, submodel_type.value):
                return getattr(pipeline, submodel_type.value)
            else:
                raise ValueError(f"Submodel {submodel_type} not found in Qwen-Image pipeline.")
        else:
            # Load the full pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=self._torch_dtype,
                variant=config.repo_variant.value if config.repo_variant else None,
            )
            return pipeline