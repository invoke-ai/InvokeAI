"""Implementation of ModelManagerServiceBase."""

from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from typing_extensions import Self

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.download.download_base import DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.model_install_base import ModelInstallServiceBase
from invokeai.app.services.model_install.model_install_default import ModelInstallService
from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.app.services.model_load.model_load_common import distinct_caches
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase
from invokeai.app.services.model_records.model_records_base import (
    InvalidModelException,
    ModelRecordChanges,
    ModelRecordServiceBase,
)
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


class ModelManagerService(ModelManagerServiceBase):
    """
    The ModelManagerService handles various aspects of model installation, maintenance and loading.

    It bundles three distinct services:
    model_manager.store   -- Routines to manage the database of model configuration records.
    model_manager.install -- Routines to install, move and delete models.
    model_manager.load    -- Routines to load models into memory.
    """

    def __init__(
        self,
        store: ModelRecordServiceBase,
        install: ModelInstallServiceBase,
        load: ModelLoadServiceBase,
    ):
        self._store = store
        self._install = install
        self._load = load

    @property
    def store(self) -> ModelRecordServiceBase:
        return self._store

    @property
    def install(self) -> ModelInstallServiceBase:
        return self._install

    @property
    def load(self) -> ModelLoadServiceBase:
        return self._load

    def update_model_record(self, key: str, changes: ModelRecordChanges) -> AnyModelConfig:
        return self._load.apply_record_change(
            key, lambda: self._store.update_model(key, changes=changes, allow_class_change=True)
        )

    def reidentify_model(self, key: str) -> AnyModelConfig:
        config = self._store.get_model(key)
        models_path = self._install.app_config.models_path
        if Path(config.path).is_relative_to(models_path):
            model_path = Path(config.path)
        else:
            model_path = models_path / config.path
        mod = ModelOnDisk(model_path)
        # The install source is evidence identification reads: for a model whose weights cannot name its
        # backbone (a 16-channel VAE, a PiD decoder), the HF repo or URL may. A file copied into the models
        # folder has also lost the folder it came from, so without the source a re-probe refiles it under
        # the default.
        override_fields = {"source": config.source, "source_type": config.source_type}
        if encoder_id := getattr(config, "image_encoder_model_id", None):
            override_fields["image_encoder_model_id"] = encoder_id
        result = ModelConfigFactory.from_model_on_disk(mod, override_fields)
        if result.config is None:
            raise InvalidModelException("Unable to identify model format")

        # Retain user-editable fields from the original config
        identified = result.config
        identified.path = config.path
        identified.key = config.key
        identified.name = config.name
        identified.description = config.description
        identified.cover_image = config.cover_image
        if hasattr(identified, "trigger_phrases") and hasattr(config, "trigger_phrases"):
            identified.trigger_phrases = config.trigger_phrases
        identified.source = config.source
        identified.source_type = config.source_type
        if hasattr(config, "cpu_only") and hasattr(identified, "cpu_only"):
            identified.cpu_only = config.cpu_only
        previous_settings = getattr(config, "default_settings", None)
        updated_settings = getattr(identified, "default_settings", None)
        if previous_settings is not None and updated_settings is not None:
            # The probe may hand back an architecture's shared defaults object; copy it before writing to it.
            updated_settings = deepcopy(updated_settings)
            identified.default_settings = updated_settings
            for field in ("cpu_only", "fp8_storage"):
                if hasattr(previous_settings, field) and hasattr(updated_settings, field):
                    setattr(updated_settings, field, getattr(previous_settings, field))

        # The probe can change the config class or reset load-affecting settings.
        return self._load.apply_record_change(config.key, lambda: self._store.replace_model(config.key, identified))

    def start(self, invoker: Invoker) -> None:
        for service in [self._store, self._install, self._load]:
            if hasattr(service, "start"):
                service.start(invoker)

    def stop(self, invoker: Invoker) -> None:
        # Shutdown every per-device model cache to cancel any pending keep-alive timers.
        if hasattr(self._load, "ram_caches"):
            for cache in distinct_caches(self._load.ram_caches.values()):
                cache.shutdown()

        for service in [self._store, self._install, self._load]:
            if hasattr(service, "stop"):
                service.stop(invoker)

    @classmethod
    def build_model_manager(
        cls,
        app_config: InvokeAIAppConfig,
        model_record_service: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        events: EventServiceBase,
        execution_device: Optional[torch.device] = None,
    ) -> Self:
        """
        Construct the model manager service instance.

        For simplicity, use this class method rather than the __init__ constructor.
        """
        logger = InvokeAILogger.get_logger(cls.__name__)
        logger.setLevel(app_config.log_level.upper())

        # One store + budget shared by every per-device cache. The store deduplicates each model's CPU
        # weights to a single copy across GPUs (see SharedCpuWeightsStore); the budget is the single
        # system-wide RAM authority so per-device caches stop double-counting shared weights when they
        # decide what to evict (see RamBudget).
        shared_store = SharedCpuWeightsStore()

        def build_cache(device: torch.device) -> ModelCache:
            return ModelCache(
                execution_device_working_mem_gb=app_config.device_working_mem_gb,
                enable_partial_loading=app_config.enable_partial_loading,
                keep_ram_copy_of_weights=app_config.keep_ram_copy_of_weights,
                max_ram_cache_size_gb=app_config.max_cache_ram_gb,
                max_vram_cache_size_gb=app_config.max_cache_vram_gb,
                reserve_vram_gb=app_config.reserve_vram_gb,
                execution_device=device,
                storage_device="cpu",
                log_memory_usage=app_config.log_memory_usage,
                logger=logger,
                keep_alive_minutes=app_config.model_cache_keep_alive_min,
                shared_cpu_weights=shared_store,
            )

        # The default cache for callers without a pinned device (API threads, single-device installs).
        default_device = execution_device or TorchDevice.choose_torch_device()
        ram_cache = build_cache(default_device)

        # In multi-GPU mode, build one independent cache per generation device. Each session-processor
        # worker is pinned to a device (see TorchDevice.set_session_device) and resolves to its own
        # cache. The default cache is always included by ModelLoadService.
        ram_caches: dict[str, ModelCache] = {str(TorchDevice.normalize(default_device)): ram_cache}
        for device in TorchDevice.get_generation_devices(app_config.generation_devices):
            key = str(device)
            if key not in ram_caches:
                ram_caches[key] = build_cache(device)

        # Attach the single global RAM budget. The cap is the user's max_cache_ram_gb interpreted as a
        # true system-wide limit; when unset, it is the sum of the caches' individually-calculated
        # sizes, so each device keeps its effective capacity and weight deduplication becomes headroom.
        # That sum is then clamped to a safe fraction of system RAM: each per-device heuristic already
        # allows up to ~half of system RAM, so summing across N GPUs would otherwise claim ~N× that and
        # leave nothing for the OS, causing swap thrashing. The clamp leaves real headroom; shared-weight
        # dedup means the true footprint usually stays well under the cap regardless.
        gb = 2**30
        device_caches = distinct_caches(ram_caches.values())
        # Cross-device weight adoption (and its per-model meta-shell capture) only pays off with more
        # than one device cache; disable the capture cost otherwise.
        shared_store.enable_shell_capture = len(device_caches) > 1
        if app_config.max_cache_ram_gb is not None:
            global_ram_budget_bytes = int(app_config.max_cache_ram_gb * gb)
        else:
            summed_cache_bytes = sum(c.local_ram_cache_size_bytes for c in device_caches)
            system_ram_headroom_bytes = ModelCache.calc_system_ram_headroom_bytes()
            global_ram_budget_bytes = min(summed_cache_bytes, system_ram_headroom_bytes)
            if global_ram_budget_bytes < summed_cache_bytes:
                logger.info(
                    f"Capping model cache RAM budget at {global_ram_budget_bytes / gb:.2f} GB to leave system "
                    f"headroom (sum of per-device caches was {summed_cache_bytes / gb:.2f} GB)."
                )
        ram_budget = RamBudget(max_bytes=global_ram_budget_bytes, shared_store=shared_store)
        for cache in device_caches:
            cache.set_ram_budget(ram_budget)
        logger.info(
            f"Model cache global RAM budget: {global_ram_budget_bytes / gb:.2f} GB "
            f"across {len(device_caches)} device cache(s)."
        )

        loader = ModelLoadService(
            app_config=app_config,
            ram_cache=ram_cache,
            registry=ModelLoaderRegistry,
            ram_caches=ram_caches,
        )
        installer = ModelInstallService(
            app_config=app_config,
            record_store=model_record_service,
            download_queue=download_queue,
            event_bus=events,
        )
        return cls(store=model_record_service, install=installer, load=loader)
