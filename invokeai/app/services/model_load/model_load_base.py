"""Base class for model loader."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Optional

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load import LoadedModel, LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType


class ModelLoadServiceBase(ABC):
    """Wrapper around AnyModelLoader."""

    @abstractmethod
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

    @abstractmethod
    def apply_record_change(self, key: str, write: Callable[[], AnyModelConfig]) -> AnyModelConfig:
        """Run `write`, which commits a change to `key`'s record and returns the record as stored, and evict
        what the change made stale; return what `write` returned.

        Every write of a model record that may change how the model loads must go through here. The stored
        record is read before `write` runs; when the two differ in anything that loads (`load_settings_changed`,
        paths compared after resolving them against the models directory), every cached entry of the model is
        dropped from each device cache. Cold loads of `key` that start meanwhile wait for the change to end
        before reading the record, and a load that read it earlier is rejected, and retried from the stored
        record, if the change affected loading. If `write` raises an `Exception`, it may have committed first, so the
        change is judged the same way by the record read back from the store before the exception propagates; if
        that read fails too, the change counts as load-affecting. An interrupt (a `BaseException` that is not an
        `Exception`) propagates without the read-back or any drop, but loads that overlapped are still re-checked.
        The caller must not hold MODEL_LOAD_LOCK."""

    @abstractmethod
    def drop_cached_models(self, keys: Iterable[str]) -> int:
        """Drop every cached entry of the models `keys` from each device cache, so their next loads rebuild;
        return how many entries were dropped at once (in-use entries are marked stale and evicted on release).

        Takes MODEL_LOAD_LOCK's write lock, so no load in construction can re-register state it built from the
        models' previous records around the drop. The caller must not hold that lock."""

    @property
    @abstractmethod
    def ram_cache(self) -> ModelCache:
        """Return the RAM cache for the current thread's execution device.

        In multi-GPU mode, each session-processor worker is pinned to a device and gets its own
        cache; this resolves to the calling thread's cache. Outside a worker (e.g. API threads),
        it resolves to the default device's cache.
        """

    @property
    @abstractmethod
    def ram_caches(self) -> dict[str, ModelCache]:
        """Return all per-device RAM caches, keyed by normalized device string.

        Use this for maintenance operations that must apply to every device (clear cache, drop a
        model from all devices, shutdown).
        """

    @abstractmethod
    def load_model_from_path(
        self, model_path: Path, loader: Optional[Callable[[Path], AnyModel]] = None
    ) -> LoadedModelWithoutConfig:
        """
        Load the model file or directory located at the indicated Path.

        This will load an arbitrary model file into the RAM cache. If the optional loader
        argument is provided, the loader will be invoked to load the model into
        memory. Otherwise the method will call safetensors.torch.load_file() or
        torch.load() as appropriate to the file suffix.

        Be aware that this returns a LoadedModelWithoutConfig object, which is the same as
        LoadedModel, but without the config attribute.

        Args:
          model_path: A pathlib.Path to a checkpoint-style models file
          loader: A Callable that expects a Path and returns a Dict[str, Tensor]

        Returns:
          A LoadedModel object.
        """
