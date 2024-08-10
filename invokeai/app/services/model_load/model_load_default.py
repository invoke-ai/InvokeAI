# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Team
"""Implementation of model loader service."""

from pathlib import Path
from typing import Callable, Optional, Type

from picklescan.scanner import scan_file_path
from safetensors.torch import load_file as safetensors_load_file
from torch import load as torch_load

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.backend.model_manager import AnyModel, AnyModelConfig, SubModelType
from invokeai.backend.model_manager.load import (
    LoadedModel,
    LoadedModelWithoutConfig,
    ModelLoaderRegistry,
    ModelLoaderRegistryBase,
)
from invokeai.backend.model_manager.load.model_cache.model_cache_base import ModelCacheBase
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


class ModelLoadService(ModelLoadServiceBase):
    """Wrapper around ModelLoaderRegistry."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        ram_cache: ModelCacheBase[AnyModel],
        registry: Optional[Type[ModelLoaderRegistryBase]] = ModelLoaderRegistry,
    ):
        """Initialize the model load service."""
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.setLevel(app_config.log_level.upper())
        self._logger = logger
        self._app_config = app_config
        self._ram_cache = ram_cache
        self._registry = registry

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    @property
    def ram_cache(self) -> ModelCacheBase[AnyModel]:
        """Return the RAM cache used by this loader."""
        return self._ram_cache

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        """

        # We don't have an invoker during testing
        # TODO(psyche): Mock this method on the invoker in the tests
        if hasattr(self, "_invoker"):
            self._invoker.services.events.emit_model_load_started(model_config, submodel_type)

        implementation, model_config, submodel_type = self._registry.get_implementation(model_config, submodel_type)  # type: ignore
        loaded_model: LoadedModel = implementation(
            app_config=self._app_config,
            logger=self._logger,
            ram_cache=self._ram_cache,
        ).load_model(model_config, submodel_type)

        if hasattr(self, "_invoker"):
            self._invoker.services.events.emit_model_load_complete(model_config, submodel_type)

        return loaded_model

    def load_model_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], AnyModel]] = None
    ) -> LoadedModelWithoutConfig:
        cache_key = str(model_path)
        ram_cache = self.ram_cache
        try:
            return LoadedModelWithoutConfig(_locker=ram_cache.get(key=cache_key))
        except IndexError:
            pass

        def torch_load_file(checkpoint: Path) -> AnyModel:
            scan_result = scan_file_path(checkpoint)
            if scan_result.infected_files != 0:
                raise Exception("The model at {checkpoint} is potentially infected by malware. Aborting load.")
            result = torch_load(checkpoint, map_location="cpu")
            return result

        def diffusers_load_directory(directory: Path) -> AnyModel:
            load_class = GenericDiffusersLoader(
                app_config=self._app_config,
                logger=self._logger,
                ram_cache=self._ram_cache,
                convert_cache=self.convert_cache,
            ).get_hf_load_class(directory)
            return load_class.from_pretrained(model_path, torch_dtype=TorchDevice.choose_torch_dtype())

        loader = loader or (
            diffusers_load_directory
            if model_path.is_dir()
            else torch_load_file
            if model_path.suffix.endswith((".ckpt", ".pt", ".pth", ".bin"))
            else lambda path: safetensors_load_file(path, device="cpu")
        )
        assert loader is not None
        raw_model = loader(model_path)
        ram_cache.put(key=cache_key, model=raw_model)
        return LoadedModelWithoutConfig(_locker=ram_cache.get(key=cache_key))
