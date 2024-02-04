# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""
Base class for model loading in InvokeAI.

Use like this:

  loader = AnyModelLoader(...)
  loaded_model = loader.get_model('019ab39adfa1840455')
  with loaded_model as model:  # context manager moves model into VRAM
       # do something with loaded_model
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.config import VaeCheckpointConfig, VaeDiffusersConfig
from invokeai.backend.model_manager.load.convert_cache.convert_cache_base import ModelConvertCacheBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase, ModelLockerBase


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
        return self.locker.model


class ModelLoaderBase(ABC):
    """Abstract base class for loading models into RAM/VRAM."""

    @abstractmethod
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the loader."""
        pass

    @abstractmethod
    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its confguration.

        Given a model identified in the model configuration backend,
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

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize AnyModelLoader with its dependencies."""
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._convert_cache = convert_cache

    @property
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the RAM cache associated used by the loaders."""
        return self._ram_cache

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its configuration.

        :param key: model key, as known to the config backend
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        implementation, model_config, submodel_type = self.__class__.get_implementation(model_config, submodel_type)
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
    def get_implementation(
        cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]
    ) -> Tuple[Type[ModelLoaderBase], AnyModelConfig, Optional[SubModelType]]:
        """Get subclass of ModelLoaderBase registered to handle base and type."""
        # We have to handle VAE overrides here because this will change the model type and the corresponding implementation returned
        conf2, submodel_type = cls._handle_subtype_overrides(config, submodel_type)

        key1 = cls._to_registry_key(conf2.base, conf2.type, conf2.format)  # for a specific base type
        key2 = cls._to_registry_key(BaseModelType.Any, conf2.type, conf2.format)  # with wildcard Any
        implementation = cls._registry.get(key1) or cls._registry.get(key2)
        if not implementation:
            raise NotImplementedError(
                f"No subclass of LoadedModel is registered for base={config.base}, type={config.type}, format={config.format}"
            )
        return implementation, conf2, submodel_type

    @classmethod
    def _handle_subtype_overrides(
        cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]
    ) -> Tuple[AnyModelConfig, Optional[SubModelType]]:
        if submodel_type == SubModelType.Vae and hasattr(config, "vae") and config.vae is not None:
            model_path = Path(config.vae)
            config_class = (
                VaeCheckpointConfig if model_path.suffix in [".pt", ".safetensors", ".ckpt"] else VaeDiffusersConfig
            )
            hash = hashlib.md5(model_path.as_posix().encode("utf-8")).hexdigest()
            new_conf = config_class(path=model_path.as_posix(), name=model_path.stem, base=config.base, key=hash)
            submodel_type = None
        else:
            new_conf = config
        return new_conf, submodel_type

    @classmethod
    def register(
        cls, type: ModelType, format: ModelFormat, base: BaseModelType = BaseModelType.Any
    ) -> Callable[[Type[ModelLoaderBase]], Type[ModelLoaderBase]]:
        """Define a decorator which registers the subclass of loader."""

        def decorator(subclass: Type[ModelLoaderBase]) -> Type[ModelLoaderBase]:
            print("DEBUG: Registering class", subclass.__name__)
            key = cls._to_registry_key(base, type, format)
            cls._registry[key] = subclass
            return subclass

        return decorator
