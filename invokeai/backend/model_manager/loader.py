# Copyright (c) 2023, Lincoln D. Stein
"""Model loader for InvokeAI."""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util import Chdir, InvokeAILogger, choose_precision, choose_torch_device

from .cache import CacheStats, ModelCache, ModelLocker
from .config import BaseModelType, ModelConfigBase, ModelType, SubModelType
from .download import DownloadEventHandler
from .install import ModelInstall, ModelInstallBase
from .models import MODEL_CLASSES, InvalidModelException, ModelBase
from .storage import ModelConfigStore, get_config_store


@dataclass
class ModelInfo:
    """This is a context manager object that is used to intermediate access to a model."""

    context: ModelLocker
    name: str
    base_model: BaseModelType
    type: ModelType
    key: str
    location: Union[Path, str]
    precision: torch.dtype
    _cache: Optional[ModelCache] = None

    def __enter__(self):
        """Context entry."""
        return self.context.__enter__()

    def __exit__(self, *args, **kwargs):
        """Context exit."""
        self.context.__exit__(*args, **kwargs)


class ModelLoadBase(ABC):
    """Abstract base class for a model loader which works with the ModelConfigStore backend."""

    @abstractmethod
    def get_model(self, key: str, submodel_type: Optional[SubModelType] = None) -> ModelInfo:
        """
        Return a model given its key.

        Given a model key identified in the model configuration backend,
        return a ModelInfo object that can be used to retrieve the model.

        :param key: model key, as known to the config backend
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        pass

    @property
    @abstractmethod
    def store(self) -> ModelConfigStore:
        """Return the ModelConfigStore object that supports this loader."""
        pass

    @property
    @abstractmethod
    def installer(self) -> ModelInstallBase:
        """Return the ModelInstallBase object that supports this loader."""
        pass

    @property
    @abstractmethod
    def logger(self) -> InvokeAILogger:
        """Return the current logger."""
        pass

    @property
    def config(self) -> InvokeAIAppConfig:
        """Return the config object used by this installer."""
        pass

    @property
    @abstractmethod
    def queue(self) -> str:
        """Return the download queue object used by this object."""
        pass

    @abstractmethod
    def collect_cache_stats(self, cache_stats: CacheStats):
        """Replace cache statistics."""
        pass

    @abstractmethod
    def resolve_model_path(self, path: Union[Path, str]) -> Path:
        """Turn a potentially relative path into an absolute one in the models_dir."""
        pass

    @property
    @abstractmethod
    def precision(self) -> str:
        """Return 'float32' or 'float16'."""
        pass

    @abstractmethod
    def sync_to_config(self):
        """
        Reinitialize the store to sync in-memory and in-disk
        versions.
        """
        pass


class ModelLoad(ModelLoadBase):
    """Implementation of ModelLoadBase."""

    _app_config: InvokeAIAppConfig
    _store: ModelConfigStore
    _installer: ModelInstallBase
    _cache: ModelCache
    _logger: InvokeAILogger
    _cache_keys: dict
    _models_file: Path

    def __init__(
        self,
        config: InvokeAIAppConfig,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ):
        """
        Initialize ModelLoad object.

        :param config: The app's InvokeAIAppConfig object.
        """
        if config.model_conf_path and config.model_conf_path.exists():
            models_file = config.model_conf_path
        else:
            models_file = config.root_path / "configs/models3.yaml"
        store = get_config_store(models_file)
        if not store:
            raise ValueError(f"Invalid model configuration file: {models_file}")

        self._app_config = config
        self._store = store
        self._logger = InvokeAILogger.getLogger()
        self._installer = ModelInstall(
            store=self._store,
            logger=self._logger,
            config=self._app_config,
            event_handlers=event_handlers,
        )
        self._cache_keys = dict()
        self._models_file = models_file
        device = torch.device(choose_torch_device())
        device_name = torch.cuda.get_device_name() if device == torch.device("cuda") else ""
        precision = choose_precision(device) if config.precision == "auto" else config.precision
        dtype = torch.float32 if precision == "float32" else torch.float16

        self._logger.info(f"Using models database {models_file}")
        self._logger.info(f"Rendering device = {device} ({device_name})")
        self._logger.info(f"Maximum RAM cache size: {config.ram_cache_size}")
        self._logger.info(f"Maximum VRAM cache size: {config.vram_cache_size}")
        self._logger.info(f"Precision: {precision}")

        self._cache = ModelCache(
            max_cache_size=config.ram_cache_size,
            max_vram_cache_size=config.vram_cache_size,
            lazy_offloading=config.lazy_offload,
            execution_device=device,
            precision=dtype,
            sequential_offload=config.sequential_guidance,
            logger=self._logger,
        )

        self._scan_models_directory()

    @property
    def store(self) -> ModelConfigStore:
        """Return the ModelConfigStore instance used by this class."""
        return self._store

    @property
    def precision(self) -> str:
        """Return 'float32' or 'float16'."""
        return self._cache.precision

    @property
    def installer(self) -> ModelInstallBase:
        """Return the ModelInstallBase instance used by this class."""
        return self._installer

    @property
    def logger(self) -> InvokeAILogger:
        """Return the current logger."""
        return self._logger

    @property
    def config(self) -> InvokeAIAppConfig:
        """Return the config object used by the installer."""
        return self._app_config

    @property
    def queue(self) -> str:
        """Return the download queue object used by this object."""
        return self._installer.queue

    def get_model(self, key: str, submodel_type: Optional[SubModelType] = None) -> ModelInfo:
        """
        Get the ModelInfo corresponding to the model with key "key".

        Given a model key identified in the model configuration backend,
        return a ModelInfo object that can be used to retrieve the model.

        :param key: model key, as known to the config backend
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        model_config = self.store.get_model(key)  # May raise a UnknownModelException
        if model_config.model_type == "main" and not submodel_type:
            raise InvalidModelException("submodel_type is required when loading a main model")

        submodel_type = SubModelType(submodel_type) if submodel_type else None

        model_path, is_submodel_override = self._get_model_path(model_config, submodel_type)

        if is_submodel_override:
            model_type = submodel_type
            submodel_type = None

        model_class = self._get_implementation(model_config.base_model, model_config.model_type)
        if not model_path.exists():
            raise InvalidModelException(f"Files for model '{key}' not found at {model_path}")

        dst_convert_path = self._get_model_cache_path(model_path)
        model_path = model_class.convert_if_required(
            model_config=model_config,
            output_path=dst_convert_path,
        )

        model_context = self._cache.get_model(
            model_path=model_path,
            model_class=model_class,
            base_model=model_config.base_model,
            model_type=model_config.model_type,
            submodel=submodel_type,
        )

        if key not in self._cache_keys:
            self._cache_keys[key] = set()
        self._cache_keys[key].add(model_context.key)

        return ModelInfo(
            context=model_context,
            name=model_config.name,
            base_model=model_config.base_model,
            type=submodel_type or model_config.model_type,
            key=model_config.key,
            location=model_path,
            precision=self._cache.precision,
            _cache=self._cache,
        )

    def collect_cache_stats(self, cache_stats: CacheStats):
        self._cache.stats = cache_stats

    def resolve_model_path(self, path: Union[Path, str]) -> Path:
        """Turn a potentially relative path into an absolute one in the models_dir."""
        return self._app_config.models_path / path

    def _get_implementation(self, base_model: BaseModelType, model_type: ModelType) -> type[ModelBase]:
        """Get the concrete implementation class for a specific model type."""
        model_class = MODEL_CLASSES[base_model][model_type]
        return model_class

    def _get_model_cache_path(self, model_path):
        return self.resolve_model_path(Path(".cache") / hashlib.md5(str(model_path).encode()).hexdigest())

    def _get_model_path(
        self, model_config: ModelConfigBase, submodel_type: Optional[SubModelType] = None
    ) -> (Path, bool):
        """Extract a model's filesystem path from its config.

        :return: The fully qualified Path of the module (or submodule).
        """
        model_path = model_config.path
        is_submodel_override = False

        # Does the config explicitly override the submodel?
        if submodel_type is not None and hasattr(model_config, submodel_type):
            submodel_path = getattr(model_config, submodel_type)
            if submodel_path is not None and len(submodel_path) > 0:
                model_path = getattr(model_config, submodel_type)
                is_submodel_override = True

        model_path = self.resolve_model_path(model_path)
        return model_path, is_submodel_override

    def sync_to_config(self):
        self._store = get_config_store(self._models_file)
        self._scan_models_directory()

    def _scan_models_directory(self):
        defunct_models = set()
        installed = set()

        with Chdir(self._app_config.models_path):
            self._logger.info("Checking for models that have been moved or deleted from disk.")
            for model_config in self._store.all_models():
                path = self.resolve_model_path(model_config.path)
                if not path.exists():
                    self._logger.info(f"{model_config.name}: path {path.as_posix()} no longer exists. Unregistering.")
                    defunct_models.add(model_config.key)
            for key in defunct_models:
                self._installer.unregister(key)

            self._logger.info(f"Scanning {self._app_config.models_path} for new models")
            for cur_base_model in BaseModelType:
                for cur_model_type in ModelType:
                    models_dir = self.resolve_model_path(Path(cur_base_model.value, cur_model_type.value))
                    installed.update(self._installer.scan_directory(models_dir))
            self._logger.info(f"{len(installed)} new models registered; {len(defunct_models)} unregistered")
