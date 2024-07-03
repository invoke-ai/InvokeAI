# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development team
from typing import Optional, Tuple, Type

from invokeai.backend.model_manager.config import BaseModelType, ModelConfigBase, ModelFormat, ModelType
from invokeai.backend.model_manager.load.load_base import AnyModelConfig, ModelLoaderBase, SubModelType


class ModelLoaderRegistry:
    """A registry that tracks which model loader class to use for a given model type/format/base combination."""

    def __init__(self):
        self._registry: dict[str, Type[ModelLoaderBase]] = {}

    def register(
        self,
        loader_class: Type[ModelLoaderBase],
        type: ModelType,
        format: ModelFormat,
        base: BaseModelType = BaseModelType.Any,
    ):
        """Register a model loader class."""
        key = self._to_registry_key(base, type, format)
        if key in self._registry:
            raise RuntimeError(
                f"{loader_class.__name__} is trying to register as a loader for {base}/{type}/{format}, but this type "
                f"of model has already been registered by {self._registry[key].__name__}"
            )
        self._registry[key] = loader_class

    def get_implementation(
        self, config: AnyModelConfig, submodel_type: Optional[SubModelType]
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

        key1 = self._to_registry_key(config.base, config.type, config.format)  # for a specific base type
        key2 = self._to_registry_key(BaseModelType.Any, config.type, config.format)  # with wildcard Any
        implementation = self._registry.get(key1, None) or self._registry.get(key2, None)
        if not implementation:
            raise NotImplementedError(
                f"No subclass of ModelLoaderBase is registered for base={config.base}, type={config.type}, "
                f"format={config.format}"
            )
        return implementation, config, submodel_type

    @staticmethod
    def _to_registry_key(base: BaseModelType, type: ModelType, format: ModelFormat) -> str:
        return "-".join([base.value, type.value, format.value])
