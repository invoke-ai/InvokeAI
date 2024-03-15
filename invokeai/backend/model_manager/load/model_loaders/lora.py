# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for LoRA model loading in InvokeAI."""

from logging import Logger
from pathlib import Path
from typing import Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.lora import LoRAModelRaw
from invokeai.backend.model_manager import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.any_model_type import AnyModel
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase

from .. import ModelLoader, ModelLoaderRegistry


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.LoRA, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.LoRA, format=ModelFormat.LyCORIS)
class LoRALoader(ModelLoader):
    """Class to load LoRA models."""

    # We cheat a little bit to get access to the model base
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the loader."""
        super().__init__(app_config, logger, ram_cache, convert_cache)
        self._model_base: Optional[BaseModelType] = None

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("There are no submodels in a LoRA model.")
        model_path = Path(config.path)
        assert self._model_base is not None
        model = LoRAModelRaw.from_checkpoint(
            file_path=model_path,
            dtype=self._torch_dtype,
            base_model=self._model_base,
        )
        return model

    # override
    def _get_model_path(self, config: AnyModelConfig) -> Path:
        # cheating a little - we remember this variable for using in the subsequent call to _load_model()
        self._model_base = config.base

        model_base_path = self._app_config.models_path
        model_path = model_base_path / config.path

        if config.format == ModelFormat.Diffusers:
            for ext in ["safetensors", "bin"]:  # return path to the safetensors file inside the folder
                path = model_base_path / config.path / f"pytorch_lora_weights.{ext}"
                if path.exists():
                    model_path = path
                    break

        return model_path.resolve()
