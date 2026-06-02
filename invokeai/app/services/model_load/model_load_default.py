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
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load import (
    LoadedModel,
    LoadedModelWithoutConfig,
    ModelLoaderRegistry,
    ModelLoaderRegistryBase,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK, ModelCache
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


class ModelLoadService(ModelLoadServiceBase):
    """Wrapper around ModelLoaderRegistry."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        ram_cache: ModelCache,
        registry: Optional[Type[ModelLoaderRegistryBase]] = ModelLoaderRegistry,
        ram_caches: Optional[dict[str, ModelCache]] = None,
    ):
        """Initialize the model load service.

        Args:
            ram_cache: The default RAM cache, used when no per-device cache matches the calling
                thread (e.g. single-device installs, or API threads).
            ram_caches: Optional map of normalized device string -> ModelCache for multi-GPU mode.
                One cache per generation device. The default `ram_cache` is always included.
        """
        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.setLevel(app_config.log_level.upper())
        self._logger = logger
        self._app_config = app_config
        self._default_ram_cache = ram_cache
        # Map normalized device string -> cache. Always includes the default cache so that callers
        # without a pinned device (API threads) resolve to a valid cache.
        self._ram_caches: dict[str, ModelCache] = dict(ram_caches) if ram_caches else {}
        self._ram_caches.setdefault(str(TorchDevice.normalize(ram_cache.execution_device)), ram_cache)
        self._registry = registry

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    @property
    def ram_cache(self) -> ModelCache:
        """Return the RAM cache for the calling thread's execution device.

        `choose_torch_device()` is thread-local-aware: a session-processor worker pinned to a GPU
        gets that GPU's cache; everything else falls back to the default cache.
        """
        key = str(TorchDevice.choose_torch_device())
        return self._ram_caches.get(key, self._default_ram_cache)

    @property
    def ram_caches(self) -> dict[str, ModelCache]:
        """Return all per-device RAM caches, keyed by normalized device string."""
        return dict(self._ram_caches)

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
            ram_cache=self.ram_cache,
        ).load_model(model_config, submodel_type)

        if hasattr(self, "_invoker"):
            self._invoker.services.events.emit_model_load_complete(model_config, submodel_type)

        return loaded_model

    def load_model_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], AnyModel]] = None
    ) -> LoadedModelWithoutConfig:
        # Resolve the calling thread's cache once so the whole load uses a single device's cache.
        ram_cache = self.ram_cache
        cache_key = str(model_path)
        try:
            return LoadedModelWithoutConfig(cache_record=ram_cache.get(key=cache_key), cache=ram_cache)
        except IndexError:
            pass

        def torch_load_file(checkpoint: Path) -> AnyModel:
            scan_result = scan_file_path(checkpoint)
            if scan_result.infected_files != 0:
                if self._app_config.unsafe_disable_picklescan:
                    self._logger.warning(
                        f"Model at {checkpoint} is potentially infected by malware, but picklescan is disabled. "
                        "Proceeding with caution."
                    )
                else:
                    raise Exception(f"The model at {checkpoint} is potentially infected by malware. Aborting load.")
            if scan_result.scan_err:
                if self._app_config.unsafe_disable_picklescan:
                    self._logger.warning(
                        f"Error scanning model at {checkpoint} for malware, but picklescan is disabled. "
                        "Proceeding with caution."
                    )
                else:
                    raise Exception(f"Error scanning model at {checkpoint} for malware. Aborting load.")

            result = torch_load(checkpoint, map_location="cpu")
            return result

        def diffusers_load_directory(directory: Path) -> AnyModel:
            load_class = GenericDiffusersLoader(
                app_config=self._app_config,
                logger=self._logger,
                ram_cache=ram_cache,
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
        # Serialize construction (see MODEL_LOAD_LOCK): the diffusers loader path uses the same
        # process-global, non-thread-safe monkey-patches as the main loader, so it takes the write
        # lock to exclude concurrent VRAM moves. Re-check the cache after acquiring the lock in case
        # a worker sharing this cache built it while we waited.
        with MODEL_LOAD_LOCK.write_lock():
            try:
                return LoadedModelWithoutConfig(cache_record=ram_cache.get(key=cache_key), cache=ram_cache)
            except IndexError:
                pass
            raw_model = loader(model_path)
            ram_cache.put(key=cache_key, model=raw_model)
            return LoadedModelWithoutConfig(cache_record=ram_cache.get(key=cache_key), cache=ram_cache)
