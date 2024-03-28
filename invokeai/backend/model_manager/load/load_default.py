# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Default implementation of model loading in InvokeAI."""

from logging import Logger
from pathlib import Path
from typing import Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    InvalidModelConfigException,
    SubModelType,
)
from invokeai.backend.model_manager.config import DiffusersConfigBase, ModelType
from invokeai.backend.model_manager.load.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.load_base import LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase, ModelLockerBase
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data, calc_model_size_by_fs
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.util.devices import choose_torch_device, torch_dtype


# TO DO: The loader is not thread safe!
class ModelLoader(ModelLoaderBase):
    """Default implementation of ModelLoaderBase."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase[AnyModel],
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the loader."""
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._convert_cache = convert_cache
        self._torch_dtype = torch_dtype(choose_torch_device(), app_config)

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its configuration.

        Given a model's configuration as returned by the ModelRecordConfigStore service,
        return a LoadedModel object that can be used for inference.

        :param model config: Configuration record for this model
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        if model_config.type is ModelType.Main and not submodel_type:
            raise InvalidModelConfigException("submodel_type is required when loading a main model")

        model_path = self._get_model_path(model_config)

        if not model_path.exists():
            raise InvalidModelConfigException(f"Files for model '{model_config.name}' not found at {model_path}")

        model_path = self._convert_if_needed(model_config, model_path, submodel_type)
        locker = self._load_if_needed(model_config, model_path, submodel_type)
        return LoadedModel(config=model_config, _locker=locker)

    def _get_model_path(self, config: AnyModelConfig) -> Path:
        model_base = self._app_config.models_path
        return (model_base / config.path).resolve()

    def _convert_if_needed(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> Path:
        cache_path: Path = self._convert_cache.cache_path(config.key)

        if not self._needs_conversion(config, model_path, cache_path):
            return cache_path if cache_path.exists() else model_path

        self._convert_cache.make_room(self.get_size_fs(config, model_path, submodel_type))
        return self._convert_model(config, model_path, cache_path)

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        return False

    def _load_if_needed(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> ModelLockerBase:
        # TO DO: This is not thread safe!
        try:
            return self._ram_cache.get(config.key, submodel_type)
        except IndexError:
            pass

        self._ram_cache.make_room(self.get_size_fs(config, model_path, submodel_type))
        config.path = model_path.as_posix()

        # This is where the model is actually loaded!
        with skip_torch_weight_init():
            loaded_model = self._load_model(config, submodel_type=submodel_type)

        self._ram_cache.put(
            config.key,
            submodel_type=submodel_type,
            model=loaded_model,
            size=calc_model_size_by_data(loaded_model),
        )

        return self._ram_cache.get(
            key=config.key,
            submodel_type=submodel_type,
            stats_name=":".join([config.base, config.type, config.name, (submodel_type or "")]),
        )

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Get the size of the model on disk."""
        return calc_model_size_by_fs(
            model_path=model_path,
            subfolder=submodel_type.value if submodel_type else None,
            variant=config.repo_variant if isinstance(config, DiffusersConfigBase) else None,
        )

    # This needs to be implemented in subclasses that handle checkpoints
    def _convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Path) -> Path:
        raise NotImplementedError

    # This needs to be implemented in the subclass
    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        raise NotImplementedError
