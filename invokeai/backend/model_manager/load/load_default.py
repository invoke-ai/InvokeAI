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
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
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

        with skip_torch_weight_init():
            locker = self._convert_and_load(model_config, model_path, submodel_type)
        return LoadedModel(config=model_config, _locker=locker)

    @property
    def convert_cache(self) -> ModelConvertCacheBase:
        """Return the convert cache associated with this loader."""
        return self._convert_cache

    @property
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the ram cache associated with this loader."""
        return self._ram_cache

    def _get_model_path(self, config: AnyModelConfig) -> Path:
        model_base = self._app_config.models_path
        return (model_base / config.path).resolve()

    def _convert_and_load(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> ModelLockerBase:
        try:
            return self._ram_cache.get(config.key, submodel_type)
        except IndexError:
            pass

        cache_path: Path = self._convert_cache.cache_path(config.key)
        if self._needs_conversion(config, model_path, cache_path):
            loaded_model = self._do_convert(config, model_path, cache_path, submodel_type)
        else:
            config.path = str(cache_path) if cache_path.exists() else str(self._get_model_path(config))
            loaded_model = self._load_model(config, submodel_type)

        self._ram_cache.put(
            config.key,
            submodel_type=submodel_type,
            model=loaded_model,
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

    def _do_convert(
        self, config: AnyModelConfig, model_path: Path, cache_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> AnyModel:
        self.convert_cache.make_room(calc_model_size_by_fs(model_path))
        pipeline = self._convert_model(config, model_path, cache_path if self.convert_cache.max_size > 0 else None)
        if submodel_type:
            # Proactively load the various submodels into the RAM cache so that we don't have to re-convert
            # the entire pipeline every time a new submodel is needed.
            for subtype in SubModelType:
                if subtype == submodel_type:
                    continue
                if submodel := getattr(pipeline, subtype.value, None):
                    self._ram_cache.put(config.key, submodel_type=subtype, model=submodel)
        return getattr(pipeline, submodel_type.value) if submodel_type else pipeline

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        return False

    # This needs to be implemented in subclasses that handle checkpoints
    def _convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Optional[Path] = None) -> AnyModel:
        raise NotImplementedError

    # This needs to be implemented in the subclass
    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        raise NotImplementedError
