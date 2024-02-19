# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team
"""Implementation of ModelManagerServiceBase."""

from typing import Optional

from typing_extensions import Self

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.invocation_context import InvocationContextData
from invokeai.backend.model_manager import AnyModelConfig, BaseModelType, LoadedModel, ModelType, SubModelType
from invokeai.backend.model_manager.load import ModelCache, ModelConvertCache, ModelLoaderRegistry
from invokeai.backend.util.logging import InvokeAILogger

from ..config import InvokeAIAppConfig
from ..download import DownloadQueueServiceBase
from ..events.events_base import EventServiceBase
from ..model_install import ModelInstallService, ModelInstallServiceBase
from ..model_load import ModelLoadService, ModelLoadServiceBase
from ..model_records import ModelRecordServiceBase, UnknownModelException
from .model_manager_base import ModelManagerServiceBase


class ModelManagerService(ModelManagerServiceBase):
    """
    The ModelManagerService handles various aspects of model installation, maintenance and loading.

    It bundles three distinct services:
    model_manager.store   -- Routines to manage the database of model configuration records.
    model_manager.install -- Routines to install, move and delete models.
    model_manager.load    -- Routines to load models into memory.
    """

    def __init__(
        self,
        store: ModelRecordServiceBase,
        install: ModelInstallServiceBase,
        load: ModelLoadServiceBase,
    ):
        self._store = store
        self._install = install
        self._load = load

    @property
    def store(self) -> ModelRecordServiceBase:
        return self._store

    @property
    def install(self) -> ModelInstallServiceBase:
        return self._install

    @property
    def load(self) -> ModelLoadServiceBase:
        return self._load

    def start(self, invoker: Invoker) -> None:
        for service in [self._store, self._install, self._load]:
            if hasattr(service, "start"):
                service.start(invoker)

    def stop(self, invoker: Invoker) -> None:
        for service in [self._store, self._install, self._load]:
            if hasattr(service, "stop"):
                service.stop(invoker)

    def load_model_by_config(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
    ) -> LoadedModel:
        return self.load.load_model(model_config, submodel_type, context_data)

    def load_model_by_key(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
    ) -> LoadedModel:
        config = self.store.get_model(key)
        return self.load.load_model(config, submodel_type, context_data)

    def load_model_by_attr(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
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
        configs = self.store.search_by_attr(model_name, base_model, model_type)
        if len(configs) == 0:
            raise UnknownModelException(f"{base_model}/{model_type}/{model_name}: Unknown model")
        elif len(configs) > 1:
            raise ValueError(f"{base_model}/{model_type}/{model_name}: More than one model matches.")
        else:
            return self.load.load_model(configs[0], submodel, context_data)

    @property
    def gpu_count(self) -> int:
        """Return the number of GPUs we are using."""
        return self.load.gpu_count

    @classmethod
    def build_model_manager(
        cls,
        app_config: InvokeAIAppConfig,
        model_record_service: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        events: EventServiceBase,
    ) -> Self:
        """
        Construct the model manager service instance.

        For simplicity, use this class method rather than the __init__ constructor.
        """
        logger = InvokeAILogger.get_logger(cls.__name__)
        logger.setLevel(app_config.log_level.upper())

        ram_cache = ModelCache(
            max_cache_size=app_config.ram_cache_size, max_vram_cache_size=app_config.vram_cache_size, logger=logger
        )
        convert_cache = ModelConvertCache(
            cache_path=app_config.models_convert_cache_path, max_size=app_config.convert_cache_size
        )
        loader = ModelLoadService(
            app_config=app_config,
            ram_cache=ram_cache,
            convert_cache=convert_cache,
            registry=ModelLoaderRegistry,
        )
        installer = ModelInstallService(
            app_config=app_config,
            record_store=model_record_service,
            download_queue=download_queue,
            event_bus=events,
        )
        return cls(store=model_record_service, install=installer, load=loader)
