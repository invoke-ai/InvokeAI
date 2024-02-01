# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""
Base class for model loading in InvokeAI.

Use like this:

  loader = AnyModelLoader(...)
  loaded_model = loader.get_model('019ab39adfa1840455')
  with loaded_model as model:  # context manager moves model into VRAM
       # do something with loaded_model
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
from diffusers import DiffusionPipeline
from injector import inject

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.backend.model_manager import AnyModelConfig, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.onnx_runtime import IAIOnnxRuntimeModel
from invokeai.backend.model_manager.ram_cache import ModelCacheBase

AnyModel = Union[DiffusionPipeline, torch.nn.Module, IAIOnnxRuntimeModel]


class ModelLockerBase(ABC):
    """Base class for the model locker used by the loader."""

    @abstractmethod
    def lock(self) -> None:
        """Lock the contained model and move it into VRAM."""
        pass

    @abstractmethod
    def unlock(self) -> None:
        """Unlock the contained model, and remove it from VRAM."""
        pass

    @property
    @abstractmethod
    def model(self) -> AnyModel:
        """Return the model."""
        pass


@dataclass
class LoadedModel:
    """Context manager object that mediates transfer from RAM<->VRAM."""

    config: AnyModelConfig
    locker: ModelLockerBase

    def __enter__(self) -> AnyModel:  # I think load_file() always returns a dict
        """Context entry."""
        self.locker.lock()
        return self.model

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Context exit."""
        self.locker.unlock()

    @property
    def model(self) -> AnyModel:
        """Return the model without locking it."""
        return self.locker.model()


class ModelLoaderBase(ABC):
    """Abstract base class for loading models into RAM/VRAM."""

    @abstractmethod
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase,
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the loader."""
        pass

    @abstractmethod
    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its key.

        Given a model key identified in the model configuration backend,
        return a ModelInfo object that can be used to retrieve the model.

        :param model_config: Model configuration, as returned by ModelConfigRecordStore
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        pass

    @abstractmethod
    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Return size in bytes of the model, calculated before loading."""
        pass


# TO DO: Better name?
class AnyModelLoader:
    """This class manages the model loaders and invokes the correct one to load a model of given base and type."""

    # this tracks the loader subclasses
    _registry: Dict[str, Type[ModelLoaderBase]] = {}

    @inject
    def __init__(
        self,
        store: ModelRecordServiceBase,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase,
        convert_cache: ModelConvertCacheBase,
    ):
        """Store the provided ModelRecordServiceBase and empty the registry."""
        self._store = store
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._convert_cache = convert_cache

    def get_model(self, key: str, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its key.

        Given a model key identified in the model configuration backend,
        return a ModelInfo object that can be used to retrieve the model.

        :param key: model key, as known to the config backend
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        model_config = self._store.get_model(key)
        implementation = self.__class__.get_implementation(
            base=model_config.base, type=model_config.type, format=model_config.format
        )
        return implementation(
            app_config=self._app_config,
            logger=self._logger,
            ram_cache=self._ram_cache,
            convert_cache=self._convert_cache,
        ).load_model(model_config, submodel_type)

    @staticmethod
    def _to_registry_key(base: BaseModelType, type: ModelType, format: ModelFormat) -> str:
        return "-".join([base.value, type.value, format.value])

    @classmethod
    def get_implementation(cls, base: BaseModelType, type: ModelType, format: ModelFormat) -> Type[ModelLoaderBase]:
        """Get subclass of ModelLoaderBase registered to handle base and type."""
        key1 = cls._to_registry_key(base, type, format)  # for a specific base type
        key2 = cls._to_registry_key(BaseModelType.Any, type, format)  # with wildcard Any
        implementation = cls._registry.get(key1) or cls._registry.get(key2)
        if not implementation:
            raise NotImplementedError(
                "No subclass of LoadedModel is registered for base={base}, type={type}, format={format}"
            )
        return implementation

    @classmethod
    def register(
        cls, type: ModelType, format: ModelFormat, base: BaseModelType = BaseModelType.Any
    ) -> Callable[[Type[ModelLoaderBase]], Type[ModelLoaderBase]]:
        """Define a decorator which registers the subclass of loader."""

        def decorator(subclass: Type[ModelLoaderBase]) -> Type[ModelLoaderBase]:
            print("Registering class", subclass.__name__)
            key = cls._to_registry_key(base, type, format)
            cls._registry[key] = subclass
            return subclass

        return decorator


# in _init__.py will call something like
# def configure_loader_dependencies(binder):
#     binder.bind(ModelRecordServiceBase, ApiDependencies.invoker.services.model_records, scope=singleton)
#     binder.bind(InvokeAIAppConfig, ApiDependencies.invoker.services.configuration, scope=singleton)
# etc
# injector = Injector(configure_loader_dependencies)
# loader = injector.get(ModelFactory)
