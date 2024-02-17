# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Implementation of model loader service."""

from typing import Optional

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.invocation_processor.invocation_processor_common import CanceledException
from invokeai.app.services.model_records import ModelRecordServiceBase, UnknownModelException
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, BaseModelType, ModelType, SubModelType
from invokeai.backend.model_manager.load import AnyModelLoader, LoadedModel, ModelCache, ModelConvertCache
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase
from invokeai.backend.util.logging import InvokeAILogger

from .model_load_base import ModelLoadServiceBase


class ModelLoadService(ModelLoadServiceBase):
    """Wrapper around AnyModelLoader."""

    def __init__(
            self,
            app_config: InvokeAIAppConfig,
            record_store: ModelRecordServiceBase,
            ram_cache: ModelCacheBase[AnyModel],
            convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the model load service."""
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.setLevel(app_config.log_level.upper())
        self._store = record_store
        self._any_loader = AnyModelLoader(
            app_config=app_config,
            logger=logger,
            ram_cache=ram_cache,
            convert_cache=convert_cache,
        )

    @property
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the RAM cache used by this loader."""
        return self._any_loader.ram_cache

    @property
    def convert_cache(self) -> ModelConvertCacheBase:
        """Return the checkpoint convert cache used by this loader."""
        return self._any_loader.convert_cache

    def load_model_by_key(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> LoadedModel:
        """
        Given a model's key, load it and return the LoadedModel object.

        :param key: Key of model config to be fetched.
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context: Invocation context used for event reporting
        """
        config = self._store.get_model(key)
        return self.load_model_by_config(config, submodel_type, context)

    def load_model_by_attr(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> LoadedModel:
        """
        Given a model's attributes, search the database for it, and if found, load and return the LoadedModel object.

        This is provided for API compatability with the get_model() method
        in the original model manager. However, note that LoadedModel is
        not the same as the original ModelInfo that ws returned.

        :param model_name: Name of to be fetched.
        :param base_model: Base model
        :param model_type: Type of the model
        :param submodel: For main (pipeline models), the submodel to fetch
        :param context: The invocation context.

        Exceptions: UnknownModelException -- model with this key not known
                    NotImplementedException -- a model loader was not provided at initialization time
                    ValueError -- more than one model matches this combination
        """
        configs = self._store.search_by_attr(model_name, base_model, model_type)
        if len(configs) == 0:
            raise UnknownModelException(f"{base_model}/{model_type}/{model_name}: Unknown model")
        elif len(configs) > 1:
            raise ValueError(f"{base_model}/{model_type}/{model_name}: More than one model matches.")
        else:
            return self.load_model_by_key(configs[0].key, submodel)

    def load_model_by_config(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param context: Invocation context used for event reporting
        """
        if context:
            self._emit_load_event(
                context=context,
                model_config=model_config,
            )
        loaded_model = self._any_loader.load_model(model_config, submodel_type)
        if context:
            self._emit_load_event(
                context=context,
                model_config=model_config,
                loaded=True,
            )
        return loaded_model

    def _emit_load_event(
        self,
        context: InvocationContext,
        model_config: AnyModelConfig,
        loaded: Optional[bool] = False,
    ) -> None:
        if context.services.queue.is_canceled(context.graph_execution_state_id):
            raise CanceledException()

        if not loaded:
            context.services.events.emit_model_load_started(
                queue_id=context.queue_id,
                queue_item_id=context.queue_item_id,
                queue_batch_id=context.queue_batch_id,
                graph_execution_state_id=context.graph_execution_state_id,
                model_config=model_config,
            )
        else:
            context.services.events.emit_model_load_completed(
                queue_id=context.queue_id,
                queue_item_id=context.queue_item_id,
                queue_batch_id=context.queue_batch_id,
                graph_execution_state_id=context.graph_execution_state_id,
                model_config=model_config,
            )
