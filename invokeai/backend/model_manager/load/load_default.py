# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Default implementation of model loading in InvokeAI."""

from logging import Logger
from pathlib import Path
from typing import Optional

import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_base import LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache, get_model_cache_key
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice


# TO DO: The loader is not thread safe!
class ModelLoader(ModelLoaderBase):
    """Default implementation of ModelLoaderBase."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCache,
    ):
        """Initialize the loader."""
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._torch_dtype = TorchDevice.choose_torch_dtype()
        self._torch_device = TorchDevice.choose_torch_device()

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its configuration.

        Given a model's configuration as returned by the ModelRecordConfigStore service,
        return a LoadedModel object that can be used for inference.

        :param model config: Configuration record for this model
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        model_path = self._get_model_path(model_config)

        if not model_path.exists():
            raise FileNotFoundError(f"Files for model '{model_config.name}' not found at {model_path}")

        with skip_torch_weight_init():
            cache_record = self._load_and_cache(model_config, submodel_type)
        return LoadedModel(config=model_config, cache_record=cache_record, cache=self._ram_cache)

    @property
    def ram_cache(self) -> ModelCache:
        """Return the ram cache associated with this loader."""
        return self._ram_cache

    def _get_model_path(self, config: AnyModelConfig) -> Path:
        model_base = self._app_config.models_path
        return (model_base / config.path).resolve()

    def _get_execution_device(
        self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None
    ) -> Optional[torch.device]:
        """Determine the execution device for a model based on its configuration.

        CPU-only execution is only applied to text encoder submodels to save VRAM while keeping
        the denoiser on GPU for performance. Conditioning tensors are moved to GPU after encoding.

        Returns:
            torch.device("cpu") if the model should run on CPU only, None otherwise (use cache default).
        """
        # Check if this is a text encoder submodel of a main model with cpu_only setting
        if hasattr(config, "default_settings") and config.default_settings is not None:
            if hasattr(config.default_settings, "cpu_only") and config.default_settings.cpu_only is True:
                # Only apply CPU execution to text encoder submodels
                if submodel_type in [SubModelType.TextEncoder, SubModelType.TextEncoder2, SubModelType.TextEncoder3]:
                    return torch.device("cpu")

        # Check if this is a standalone text encoder config with cpu_only field (T5Encoder, Qwen3Encoder, etc.)
        if hasattr(config, "cpu_only") and config.cpu_only is True:
            return torch.device("cpu")

        return None

    def _load_and_cache(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> CacheRecord:
        stats_name = ":".join([config.base, config.type, config.name, (submodel_type or "")])
        try:
            return self._ram_cache.get(key=get_model_cache_key(config.key, submodel_type), stats_name=stats_name)
        except IndexError:
            pass

        config.path = str(self._get_model_path(config))
        self._ram_cache.make_room(self.get_size_fs(config, Path(config.path), submodel_type))
        loaded_model = self._load_model(config, submodel_type)

        # Determine execution device from model config, considering submodel type
        execution_device = self._get_execution_device(config, submodel_type)

        self._ram_cache.put(
            get_model_cache_key(config.key, submodel_type),
            model=loaded_model,
            execution_device=execution_device,
        )

        return self._ram_cache.get(key=get_model_cache_key(config.key, submodel_type), stats_name=stats_name)

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Get the size of the model on disk."""
        return calc_model_size_by_fs(
            model_path=model_path,
            subfolder=submodel_type.value if submodel_type else None,
            variant=config.repo_variant if isinstance(config, Diffusers_Config_Base) else None,
        )

    # This needs to be implemented in the subclass
    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        raise NotImplementedError
