# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from pydantic import Field

from invokeai.app.models.exceptions import CanceledException
from invokeai.backend.model_manager import ModelConfigStore, SubModelType
from invokeai.backend.model_manager.cache import CacheStats
from invokeai.backend.model_manager.loader import ModelInfo, ModelLoad

from .config import InvokeAIAppConfig
from .model_record_service import ModelRecordServiceBase

if TYPE_CHECKING:
    from ..invocations.baseinvocation import InvocationContext


class ModelLoadServiceBase(ABC):
    """Load models into memory."""

    @abstractmethod
    def __init__(
        self,
        config: InvokeAIAppConfig,
        store: Union[ModelConfigStore, ModelRecordServiceBase],
    ):
        """
        Initialize a ModelLoadService

        :param config: InvokeAIAppConfig object
        :param store: ModelConfigStore object for fetching configuration information
        installation and download events will be sent to the event bus.
        """
        pass

    @abstractmethod
    def get_model(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """Retrieve the indicated model identified by key.

        :param key: Unique key returned by the ModelConfigStore module.
        :param submodel_type: Submodel to return (required for main models)
        :param context" Optional InvocationContext, used in event reporting.
        """
        pass

    @abstractmethod
    def collect_cache_stats(self, cache_stats: CacheStats):
        """Reset model cache statistics for graph with graph_id."""
        pass


# implementation
class ModelLoadService(ModelLoadServiceBase):
    """Responsible for managing models on disk and in memory."""

    _loader: ModelLoad

    def __init__(
        self,
        config: InvokeAIAppConfig,
        record_store: Union[ModelConfigStore, ModelRecordServiceBase],
    ):
        """
        Initialize a ModelLoadService.

        :param config: InvokeAIAppConfig object
        :param store: ModelRecordServiceBase or ModelConfigStore object for fetching configuration information
        installation and download events will be sent to the event bus.
        """
        self._loader = ModelLoad(config, record_store)

    def get_model(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """
        Retrieve the indicated model.

        The submodel is required when fetching a main model.
        """
        model_info: ModelInfo = self._loader.get_model(key, submodel_type)

        # we can emit model loading events if we are executing with access to the invocation context
        if context:
            self._emit_load_event(
                context=context,
                model_key=key,
                submodel=submodel_type,
                model_info=model_info,
            )

        return model_info

    def collect_cache_stats(self, cache_stats: CacheStats):
        """
        Reset model cache statistics. Is this used?
        """
        self._loader.collect_cache_stats(cache_stats)

    def _emit_load_event(
        self,
        context: InvocationContext,
        model_key: str,
        submodel: Optional[SubModelType] = None,
        model_info: Optional[ModelInfo] = None,
    ):
        if context.services.queue.is_canceled(context.graph_execution_state_id):
            raise CanceledException()

        if model_info:
            context.services.events.emit_model_load_completed(
                queue_id=context.queue_id,
                queue_item_id=context.queue_item_id,
                queue_batch_id=context.queue_batch_id,
                graph_execution_state_id=context.graph_execution_state_id,
                model_key=model_key,
                submodel=submodel,
                model_info=model_info,
            )
        else:
            context.services.events.emit_model_load_started(
                queue_id=context.queue_id,
                queue_item_id=context.queue_item_id,
                queue_batch_id=context.queue_batch_id,
                graph_execution_state_id=context.graph_execution_state_id,
                model_key=model_key,
                submodel=submodel,
            )
