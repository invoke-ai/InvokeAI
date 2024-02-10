# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Implementation of model loader service."""

from typing import Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.backend.model_manager import AnyModelConfig, SubModelType
from invokeai.backend.model_manager.load import AnyModelLoader, LoadedModel, ModelCache, ModelConvertCache
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.ram_cache import ModelCacheBase
from invokeai.backend.util.logging import InvokeAILogger

from .model_load_base import ModelLoadServiceBase


class ModelLoadService(ModelLoadServiceBase):
    """Wrapper around AnyModelLoader."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
        ram_cache: Optional[ModelCacheBase] = None,
        convert_cache: Optional[ModelConvertCacheBase] = None,
    ):
        """Initialize the model load service."""
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.setLevel(app_config.log_level.upper())
        self._store = record_store
        self._any_loader = AnyModelLoader(
            app_config=app_config,
            logger=logger,
            ram_cache=ram_cache
            or ModelCache(
                max_cache_size=app_config.ram_cache_size,
                max_vram_cache_size=app_config.vram_cache_size,
                logger=logger,
            ),
            convert_cache=convert_cache
            or ModelConvertCache(
                cache_path=app_config.models_convert_cache_path,
                max_size=app_config.convert_cache_size,
            ),
        )

    def load_model_by_key(self, key: str, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """Given a model's key, load it and return the LoadedModel object."""
        config = self._store.get_model(key)
        return self.load_model_by_config(config, submodel_type)

    def load_model_by_config(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """Given a model's configuration, load it and return the LoadedModel object."""
        return self._any_loader.load_model(config, submodel_type)
