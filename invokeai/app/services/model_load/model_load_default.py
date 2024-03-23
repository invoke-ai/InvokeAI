# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Implementation of model loader service."""

from typing import Optional, Type

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.invocation_context import InvocationContextData
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, SubModelType
from invokeai.backend.model_manager.load import (
    LoadedModel,
    ModelLoaderRegistry,
    ModelLoaderRegistryBase,
)
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase
from invokeai.backend.util.logging import InvokeAILogger

from .model_load_base import ModelLoadServiceBase


class ModelLoadService(ModelLoadServiceBase):
    """Wrapper around ModelLoaderRegistry."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
        registry: Optional[Type[ModelLoaderRegistryBase]] = ModelLoaderRegistry,
    ):
        """Initialize the model load service."""
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.setLevel(app_config.log_level.upper())
        self._logger = logger
        self._app_config = app_config
        self._ram_cache = ram_cache
        self._convert_cache = convert_cache
        self._registry = registry

    def start(self, invoker: Invoker) -> None:
        """Start the service."""
        self._invoker = invoker

    @property
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the RAM cache used by this loader."""
        return self._ram_cache

    @property
    def gpu_count(self) -> int:
        """Return the number of GPUs available for our uses."""
        return len(self._ram_cache.execution_devices)

    @property
    def convert_cache(self) -> ModelConvertCacheBase:
        """Return the checkpoint convert cache used by this loader."""
        return self._convert_cache

    def load_model(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
    ) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context: Invocation context used for event reporting
        """
        if context_data:
            self._emit_load_event(
                context_data=context_data,
                model_config=model_config,
                submodel_type=submodel_type,
            )

        implementation, model_config, submodel_type = self._registry.get_implementation(model_config, submodel_type)  # type: ignore
        loaded_model: LoadedModel = implementation(
            app_config=self._app_config,
            logger=self._logger,
            ram_cache=self._ram_cache,
            convert_cache=self._convert_cache,
        ).load_model(model_config, submodel_type)

        if context_data:
            self._emit_load_event(
                context_data=context_data,
                model_config=model_config,
                submodel_type=submodel_type,
                loaded=True,
            )
        return loaded_model

    def _emit_load_event(
        self,
        context_data: InvocationContextData,
        model_config: AnyModelConfig,
        loaded: Optional[bool] = False,
        submodel_type: Optional[SubModelType] = None,
    ) -> None:
        if not self._invoker:
            return

        if not loaded:
            self._invoker.services.events.emit_model_load_started(
                queue_id=context_data.queue_item.queue_id,
                queue_item_id=context_data.queue_item.item_id,
                queue_batch_id=context_data.queue_item.batch_id,
                graph_execution_state_id=context_data.queue_item.session_id,
                model_config=model_config,
                submodel_type=submodel_type,
            )
        else:
            self._invoker.services.events.emit_model_load_completed(
                queue_id=context_data.queue_item.queue_id,
                queue_item_id=context_data.queue_item.item_id,
                queue_batch_id=context_data.queue_item.batch_id,
                graph_execution_state_id=context_data.queue_item.session_id,
                model_config=model_config,
                submodel_type=submodel_type,
            )
