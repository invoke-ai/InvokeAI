"""Implementation of model loader service."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Iterator, Optional, Type

from picklescan.scanner import scan_file_path
from safetensors.torch import load_file as safetensors_load_file
from torch import load as torch_load

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.app.services.model_load.model_load_common import RecordEdit, load_settings_changed
from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load import (
    LoadedModel,
    LoadedModelWithoutConfig,
    ModelLoader,
    ModelLoaderRegistry,
    ModelLoaderRegistryBase,
    StaleModelConfigError,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK, ModelCache
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


@dataclass
class _KeyEdits:
    in_progress: int = 0
    # Loads between their record read and their check under MODEL_LOAD_LOCK.
    watchers: int = 0
    # Bumped as each edit that changed how the model loads ends.
    generation: int = 0


class _RecordEdits:
    """Per model key: record edits in progress, and whether one that changed how the model loads has ended since a
    load read the record. A key is tracked only while it has edits in progress or loads watching it: a load that
    starts later waits for in-progress edits, so it reads their result."""

    def __init__(self) -> None:
        self._changed = threading.Condition(threading.Lock())
        self._keys: dict[str, _KeyEdits] = {}

    def _retire_if_unused(self, key: str) -> None:
        state = self._keys[key]
        if not state.in_progress and not state.watchers:
            del self._keys[key]

    @contextmanager
    def edit(self, key: str) -> Iterator[RecordEdit]:
        record_edit = RecordEdit()
        with self._changed:
            self._keys.setdefault(key, _KeyEdits()).in_progress += 1
        try:
            yield record_edit
        finally:
            with self._changed:
                state = self._keys[key]
                if record_edit.load_affecting:
                    state.generation += 1
                state.in_progress -= 1
                if not state.in_progress:
                    self._changed.notify_all()
                self._retire_if_unused(key)

    def wait_idle(self, key: str) -> None:
        """Wait until no edit of `key` is in progress."""
        with self._changed:
            self._changed.wait_for(lambda: key not in self._keys or not self._keys[key].in_progress)

    @contextmanager
    def watch(self, key: str) -> Iterator[Callable[[], bool]]:
        """Wait until no edit of `key` is in progress, then yield a callable telling whether `key` is still unedited:
        no edit in progress, and none that changed how it loads ended since the wait."""
        with self._changed:
            self._changed.wait_for(lambda: key not in self._keys or not self._keys[key].in_progress)
            state = self._keys.setdefault(key, _KeyEdits())
            state.watchers += 1
            generation = state.generation

        def unchanged() -> bool:
            with self._changed:
                return not state.in_progress and state.generation == generation

        try:
            yield unchanged
        finally:
            with self._changed:
                state.watchers -= 1
                self._retire_if_unused(key)


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
        self._record_edits = _RecordEdits()

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

    def load_model(
        self,
        model_config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
        user_id: Optional[str] = None,
    ) -> LoadedModel:
        """
        Given a model's configuration, load it and return the LoadedModel object.

        :param model_config: Model configuration record (as returned by ModelRecordBase.get_model())
        :param submodel: For main (pipeline models), the submodel to fetch.
        :param user_id: The user whose action triggered the load, threaded into the model load
            events so they can be routed to that user's UI (defaults to the system user).
        """

        # We don't have an invoker during testing
        # TODO(psyche): Mock this method on the invoker in the tests
        if hasattr(self, "_invoker"):
            self._invoker.services.events.emit_model_load_started(model_config, submodel_type, user_id or "system")

        try:
            try:
                return self._load(model_config, submodel_type)
            except StaleModelConfigError:
                # The record changed after the caller read it: loading what it says now is what this call
                # would have done had it been made a moment later. Retried here rather than by callers so
                # every caller gets it.
                return self._load(self._current_record(model_config.key), submodel_type)
            except FileNotFoundError:
                # The same, when the model was moved and its record updated to the new path; a missing
                # file under an unchanged (or deleted) record is the caller's real error.
                current = self._moved_record(model_config)
                if current is None:
                    raise
                return self._load(current, submodel_type)
        finally:
            # Sent however the load ends, so a UI showing it as in progress can clear it.
            if hasattr(self, "_invoker"):
                self._invoker.services.events.emit_model_load_complete(model_config, submodel_type, user_id or "system")

    def _load(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType]) -> LoadedModel:
        implementation, model_config, submodel_type = self._registry.get_implementation(model_config, submodel_type)  # type: ignore
        loader = implementation(
            app_config=self._app_config,
            logger=self._logger,
            ram_cache=self.ram_cache,
        )
        if not isinstance(loader, ModelLoader):
            # Only `ModelLoader` applies the stale-config check before admitting a model to the cache.
            raise TypeError(f"{implementation!r} is not a ModelLoader, so it cannot be loaded safely")
        if hasattr(self, "_invoker"):
            loader.config_check = self._check_config
        return loader.load_model(model_config, submodel_type)

    def record_edit(self, key: str) -> ContextManager[RecordEdit]:
        return self._record_edits.edit(key)

    def _current_record(self, key: str) -> AnyModelConfig:
        """The stored record for `key`, read once no edit of it is in progress."""
        self._record_edits.wait_idle(key)
        return self._invoker.services.model_manager.store.get_model(key)

    def _moved_record(self, config: AnyModelConfig) -> Optional[AnyModelConfig]:
        """The stored record for `config`'s key if it now loads differently from `config`, else None."""
        if not hasattr(self, "_invoker"):
            return None
        try:
            current = self._current_record(config.key)
        except UnknownModelException:
            return None
        if load_settings_changed(config, current, models_path=self._app_config.models_path):
            return current
        return None

    @contextmanager
    def _check_config(self, config: AnyModelConfig) -> Iterator[Callable[[], bool]]:
        """Check `config` against its record; the yielded callable completes the check under MODEL_LOAD_LOCK.

        Nothing here touches the database under that lock, so a long DB transaction (e.g. VACUUM) stalls only
        this load. The record is read here, once no edit of it is in progress; edits bracket their commit and
        cache invalidation with `record_edit()`. An edit that commits after this read therefore started after
        the wait, so by the time construction is serialized against invalidation it is either still in
        progress or has ended; either way the load is rejected, unless it ended having changed nothing that
        loads (e.g. a rename).
        """
        with self._record_edits.watch(config.key) as unedited:
            current = self._config_is_current(config)
            yield lambda: current and unedited()

    def _config_is_current(self, config: AnyModelConfig) -> bool:
        """Whether `config` still loads the same model as the stored record for its key."""
        try:
            current = self._invoker.services.model_manager.store.get_model(config.key)
        except UnknownModelException:
            # Nothing newer to load instead, and the key can no longer be requested.
            return True
        return not load_settings_changed(config, current, models_path=self._app_config.models_path)

    def load_model_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], AnyModel]] = None
    ) -> LoadedModelWithoutConfig:
        # Resolve the calling thread's cache once so the whole load uses a single device's cache.
        ram_cache = self.ram_cache
        cache_key = str(model_path)
        try:
            # Retrieve with a first-use claim so the record cannot be evicted between the lookup
            # and the wrapper's first lock (see ModelCache.get_with_first_use_claim).
            cache_record, first_use_claim = ram_cache.get_with_first_use_claim(key=cache_key)
            return LoadedModelWithoutConfig(cache_record=cache_record, cache=ram_cache, first_use_claim=first_use_claim)
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
                cache_record, first_use_claim = ram_cache.get_with_first_use_claim(key=cache_key)
                return LoadedModelWithoutConfig(
                    cache_record=cache_record, cache=ram_cache, first_use_claim=first_use_claim
                )
            except IndexError:
                pass
            raw_model = loader(model_path)
            # The admission claim shields the record until this retrieval's own claim takes over
            # (see ModelCache.put), so nothing can evict the model still held in raw_model here.
            admission_claim = ram_cache.put(key=cache_key, model=raw_model, claim_admission=True)
            cache_record, first_use_claim = ram_cache.get_with_first_use_claim(key=cache_key)
            if admission_claim is not None:
                admission_claim.release()
            return LoadedModelWithoutConfig(cache_record=cache_record, cache=ram_cache, first_use_claim=first_use_claim)
