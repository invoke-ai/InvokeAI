# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development team
"""
This module implements a system in which model loaders register the
type, base and format of models that they know how to load.

Use like this:

  cls, model_config, submodel_type = ModelLoaderRegistry.get_implementation(model_config, submodel_type)  # type: ignore
  loaded_model = cls(
       app_config=app_config,
       logger=logger,
       ram_cache=ram_cache,
       convert_cache=convert_cache
    ).load_model(model_config, submodel_type)

"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar

from ..config import (
    AnyModelConfig,
    BaseModelType,
    ModelConfigBase,
    ModelFormat,
    ModelType,
    SubModelType,
)
from . import ModelLoaderBase


class ModelLoaderRegistryBase(ABC):
    """This class allows model loaders to register their type, base and format."""

    @classmethod
    @abstractmethod
    def register(
        cls, type: ModelType, format: ModelFormat, base: BaseModelType = BaseModelType.Any
    ) -> Callable[[Type[ModelLoaderBase]], Type[ModelLoaderBase]]:
        """Define a decorator which registers the subclass of loader."""

    @classmethod
    @abstractmethod
    def get_implementation(
        cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]
    ) -> Tuple[Type[ModelLoaderBase], ModelConfigBase, Optional[SubModelType]]:
        """
        Get subclass of ModelLoaderBase registered to handle base and type.

        Parameters:
        :param config: Model configuration record, as returned by ModelRecordService
        :param submodel_type: Submodel to fetch (main models only)
        :return: tuple(loader_class, model_config, submodel_type)

        Note that the returned model config may be different from one what passed
        in, in the event that a submodel type is provided.
        """


TModelLoader = TypeVar("TModelLoader", bound=ModelLoaderBase)


class ModelLoaderRegistry(ModelLoaderRegistryBase):
    """
    This class allows model loaders to register their type, base and format.
    """

    _registry: Dict[str, Type[ModelLoaderBase]] = {}

    @classmethod
    def register(
        cls, type: ModelType, format: ModelFormat, base: BaseModelType = BaseModelType.Any
    ) -> Callable[[Type[TModelLoader]], Type[TModelLoader]]:
        """Define a decorator which registers the subclass of loader."""

        def decorator(subclass: Type[TModelLoader]) -> Type[TModelLoader]:
            key = cls._to_registry_key(base, type, format)
            if key in cls._registry:
                raise Exception(
                    f"{subclass.__name__} is trying to register as a loader for {base}/{type}/{format}, but this type of model has already been registered by {cls._registry[key].__name__}"
                )
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def get_implementation(
        cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]
    ) -> Tuple[Type[ModelLoaderBase], ModelConfigBase, Optional[SubModelType]]:
        """Get subclass of ModelLoaderBase registered to handle base and type."""

        key1 = cls._to_registry_key(config.base, config.type, config.format)  # for a specific base type
        key2 = cls._to_registry_key(BaseModelType.Any, config.type, config.format)  # with wildcard Any
        implementation = cls._registry.get(key1) or cls._registry.get(key2)
        if not implementation:
            raise NotImplementedError(
                f"No subclass of LoadedModel is registered for base={config.base}, type={config.type}, format={config.format}"
            )
        return implementation, config, submodel_type

    @staticmethod
    def _to_registry_key(base: BaseModelType, type: ModelType, format: ModelFormat) -> str:
        return "-".join([base.value, type.value, format.value])
