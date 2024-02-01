# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Default implementation of model loading in InvokeAI."""

import sys
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from injector import inject

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager import AnyModelConfig, InvalidModelConfigException, ModelRepoVariant, SubModelType
from invokeai.backend.model_manager.convert_cache import ModelConvertCacheBase
from invokeai.backend.model_manager.load.load_base import AnyModel, LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_manager.ram_cache import ModelCacheBase, ModelLockerBase
from invokeai.backend.util.devices import choose_torch_device, torch_dtype


class ConfigLoader(ConfigMixin):
    """Subclass of ConfigMixin for loading diffusers configuration files."""

    @classmethod
    def load_config(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Load a diffusrs ConfigMixin configuration."""
        cls.config_name = kwargs.pop("config_name")
        # Diffusers doesn't provide typing info
        return super().load_config(*args, **kwargs)  # type: ignore


# TO DO: The loader is not thread safe!
class ModelLoader(ModelLoaderBase):
    """Default implementation of ModelLoaderBase."""

    @inject  # can inject instances of each of the classes in the call signature
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCacheBase,
        convert_cache: ModelConvertCacheBase,
    ):
        """Initialize the loader."""
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._convert_cache = convert_cache
        self._torch_dtype = torch_dtype(choose_torch_device())
        self._size: Optional[int] = None  # model size

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its configuration.

        Given a model's configuration as returned by the ModelRecordConfigStore service,
        return a LoadedModel object that can be used for inference.

        :param model config: Configuration record for this model
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        if model_config.type == "main" and not submodel_type:
            raise InvalidModelConfigException("submodel_type is required when loading a main model")

        model_path, is_submodel_override = self._get_model_path(model_config, submodel_type)
        if is_submodel_override:
            submodel_type = None

        if not model_path.exists():
            raise InvalidModelConfigException(f"Files for model 'model_config.name' not found at {model_path}")

        model_path = self._convert_if_needed(model_config, model_path, submodel_type)
        locker = self._load_if_needed(model_config, model_path, submodel_type)
        return LoadedModel(config=model_config, locker=locker)

    # IMPORTANT: This needs to be overridden in the StableDiffusion subclass so as to handle vae overrides
    # and submodels!!!!
    def _get_model_path(
        self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None
    ) -> Tuple[Path, bool]:
        model_base = self._app_config.models_path
        return ((model_base / config.path).resolve(), False)

    def _convert_if_needed(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> Path:
        if not self._needs_conversion(config):
            return model_path

        self._convert_cache.make_room(self._size or self.get_size_fs(config, model_path, submodel_type))
        cache_path: Path = self._convert_cache.cache_path(config.key)
        if cache_path.exists():
            return cache_path

        self._convert_model(model_path, cache_path)
        return cache_path

    def _needs_conversion(self, config: AnyModelConfig) -> bool:
        return False

    def _load_if_needed(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> ModelLockerBase:
        # TO DO: This is not thread safe!
        if self._ram_cache.exists(config.key, submodel_type):
            return self._ram_cache.get(config.key, submodel_type)

        model_variant = getattr(config, "repo_variant", None)
        self._ram_cache.make_room(self.get_size_fs(config, model_path, submodel_type))

        # This is where the model is actually loaded!
        with skip_torch_weight_init():
            loaded_model = self._load_model(model_path, model_variant=model_variant, submodel_type=submodel_type)

        self._ram_cache.put(
            config.key,
            submodel_type=submodel_type,
            model=loaded_model,
        )

        return self._ram_cache.get(config.key, submodel_type)

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Get the size of the model on disk."""
        return calc_model_size_by_fs(
            model_path=model_path,
            subfolder=submodel_type.value if submodel_type else None,
            variant=config.repo_variant if hasattr(config, "repo_variant") else None,
        )

    def _convert_model(self, model_path: Path, cache_path: Path) -> None:
        raise NotImplementedError

    def _load_model(
        self,
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        raise NotImplementedError

    def _load_diffusers_config(self, model_path: Path, config_name: str = "config.json") -> Dict[str, Any]:
        return ConfigLoader.load_config(model_path, config_name=config_name)

    # TO DO: Add exception handling
    def _hf_definition_to_type(self, module: str, class_name: str) -> ModelMixin:  # fix with correct type
        if module in ["diffusers", "transformers"]:
            res_type = sys.modules[module]
        else:
            res_type = sys.modules["diffusers"].pipelines
        result: ModelMixin = getattr(res_type, class_name)
        return result

    # TO DO: Add exception handling
    def _get_hf_load_class(self, model_path: Path, submodel_type: Optional[SubModelType] = None) -> ModelMixin:
        if submodel_type:
            config = self._load_diffusers_config(model_path, config_name="model_index.json")
            module, class_name = config[submodel_type.value]
            return self._hf_definition_to_type(module=module, class_name=class_name)
        else:
            config = self._load_diffusers_config(model_path, config_name="config.json")
            class_name = config["_class_name"]
            return self._hf_definition_to_type(module="diffusers", class_name=class_name)
