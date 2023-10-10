# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import move, rmtree
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Set, Union

from pydantic import Field
from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_record_service import ModelRecordServiceBase
from invokeai.backend import get_precision
from invokeai.backend.model_manager.config import (
    BaseModelType,
    ModelConfigBase,
    ModelFormat,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    SubModelType,
)
from invokeai.backend.model_manager.download.model_queue import (
    HTTP_RE,
    REPO_ID_WITH_OPTIONAL_SUBFOLDER_RE,
    DownloadJobRepoID,
    DownloadJobWithMetadata,
)
from invokeai.backend.model_manager.hash import FastModelHash
from invokeai.backend.model_manager.models import InvalidModelException
from invokeai.backend.model_manager.probe import ModelProbe, ModelProbeInfo
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.model_manager.storage import DuplicateModelException, ModelConfigStore
from invokeai.backend.util import Chdir, InvokeAILogger, Logger

if TYPE_CHECKING:
    from .events import EventServiceBase

from .download_manager import (
    DownloadEventHandler,
    DownloadJobBase,
    DownloadJobPath,
    DownloadQueueService,
    DownloadQueueServiceBase,
    ModelSourceMetadata,
)


class ModelInstallJob(DownloadJobBase):
    """This is a version of DownloadJobBase that has an additional slot for the model key and probe info."""

    model_key: Optional[str] = Field(
        description="After model installation, this field will hold its primary key", default=None
    )
    probe_override: Optional[Dict[str, Any]] = Field(
        description="Keys in this dict will override like-named attributes in the automatic probe info",
        default=None,
    )


class ModelInstallURLJob(DownloadJobWithMetadata, ModelInstallJob):
    """Job for installing URLs."""


class ModelInstallRepoIDJob(DownloadJobRepoID, ModelInstallJob):
    """Job for installing repo ids."""


class ModelInstallPathJob(DownloadJobPath, ModelInstallJob):
    """Job for installing local paths."""


ModelInstallEventHandler = Callable[["ModelInstallJob"], None]


class ModelInstallServiceBase(ABC):
    """Abstract base class for InvokeAI model installation."""

    @abstractmethod
    def __init__(
        self,
        config: Optional[InvokeAIAppConfig] = None,
        queue: Optional[DownloadQueueServiceBase] = None,
        store: Optional[ModelRecordServiceBase] = None,
        event_bus: Optional["EventServiceBase"] = None,
    ):
        """
        Create ModelInstallService object.

        :param store: Optional ModelConfigStore. If None passed,
        defaults to `configs/models.yaml`.
        :param config: Optional InvokeAIAppConfig. If None passed,
        uses the system-wide default app config.
        :param logger: Optional InvokeAILogger. If None passed,
        uses the system-wide default logger.
        :param download: Optional DownloadQueueServiceBase object. If None passed,
        a default queue object will be created.
        :param event_handlers: List of event handlers to pass to the queue object.
        """
        pass

    @property
    @abstractmethod
    def queue(self) -> DownloadQueueServiceBase:
        """Return the download queue used by the installer."""
        pass

    @property
    @abstractmethod
    def store(self) -> ModelRecordServiceBase:
        """Return the storage backend used by the installer."""
        pass

    @property
    @abstractmethod
    def config(self) -> InvokeAIAppConfig:
        """Return the app_config used by the installer."""
        pass

    @abstractmethod
    def register_path(self, model_path: Union[Path, str], overrides: Optional[Dict[str, Any]]) -> str:
        """
        Probe and register the model at model_path.

        :param model_path: Filesystem Path to the model.
        :param overrides: Dict of attributes that will override probed values.
        :returns id: The string ID of the registered model.
        """
        pass

    @abstractmethod
    def install_path(self, model_path: Union[Path, str], overrides: Optional[Dict[str, Any]] = None) -> str:
        """
        Probe, register and install the model in the models directory.

        This involves moving the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :param overrides: Dictionary of model probe info fields that, if present, override probed values.
        :returns id: The string ID of the installed model.
        """
        pass

    @abstractmethod
    def install_model(
        self,
        source: Union[str, Path, AnyHttpUrl],
        inplace: bool = True,
        priority: int = 10,
        variant: Optional[str] = None,
        subfolder: Optional[str] = None,
        probe_override: Optional[Dict[str, Any]] = None,
        metadata: Optional[ModelSourceMetadata] = None,
        access_token: Optional[str] = None,
    ) -> ModelInstallJob:
        """
        Download and install the indicated model.

        This will download the model located at `source`,
        probe it, and install it into the models directory.
        This call is executed asynchronously in a separate
        thread, and the returned object is a
        invokeai.backend.model_manager.download.DownloadJobBase
        object which can be interrogated to get the status of
        the download and install process. Call our `wait_for_installs()`
        method to wait for all downloads and installations to complete.

        :param source: Either a URL or a HuggingFace repo_id.
        :param inplace: If True, local paths will not be moved into
        the models directory, but registered in place (the default).
        :param variant: For HuggingFace models, this optional parameter
        specifies which variant to download (e.g. 'fp16')
        :param subfolder: When downloading HF repo_ids this can be used to
        specify a subfolder of the HF repository to download from.
        :param probe_override: Optional dict. Any fields in this dict
        will override corresponding probe fields. Use it to override
        `base_type`, `model_type`, `format`, `prediction_type` and `image_size`.
        :param metadata: Use this to override the fields 'description`,
        `author`, `tags`, `source` and `license`.

        :returns ModelInstallJob object.

        The `inplace` flag does not affect the behavior of downloaded
        models, which are always moved into the `models` directory.

        Variants recognized by HuggingFace currently are:
        1. onnx
        2. openvino
        3. fp16
        4. None (usually returns fp32 model)
        """
        pass

    @abstractmethod
    def wait_for_installs(self) -> Dict[Union[str, Path, AnyHttpUrl], Optional[str]]:
        """
        Wait for all pending installs to complete.

        This will block until all pending downloads have
        completed, been cancelled, or errored out. It will
        block indefinitely if one or more jobs are in the
        paused state.

        It will return a dict that maps the source model
        path, URL or repo_id to the ID of the installed model.
        """
        pass

    @abstractmethod
    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:
        """
        Recursively scan directory for new models and register or install them.

        :param scan_dir: Path to the directory to scan.
        :param install: Install if True, otherwise register in place.
        :returns list of IDs: Returns list of IDs of models registered/installed
        """
        pass

    @abstractmethod
    def hash(self, model_path: Union[Path, str]) -> str:
        """
        Compute and return the fast hash of the model.

        :param model_path: Path to the model on disk.
        :return str: FastHash of the model for use as an ID.
        """
        pass


class ModelInstallService(ModelInstallServiceBase):
    """Model installer class handles installation from a local path."""

    _app_config: InvokeAIAppConfig
    _logger: Logger
    _store: ModelConfigStore
    _download_queue: DownloadQueueServiceBase
    _async_installs: Dict[Union[str, Path, AnyHttpUrl], Optional[str]]
    _installed: Set[str] = Field(default=set)
    _tmpdir: Optional[tempfile.TemporaryDirectory]  # used for downloads
    _cached_model_paths: Set[Path] = Field(default=set)  # used to speed up directory scanning
    _precision: Literal["float16", "float32"] = Field(description="Floating point precision, string form")
    _event_bus: Optional["EventServiceBase"] = Field(description="an event bus to send install events to", default=None)

    _legacy_configs: Dict[BaseModelType, Dict[ModelVariantType, Union[str, dict]]] = {
        BaseModelType.StableDiffusion1: {
            ModelVariantType.Normal: "v1-inference.yaml",
            ModelVariantType.Inpaint: "v1-inpainting-inference.yaml",
        },
        BaseModelType.StableDiffusion2: {
            ModelVariantType.Normal: {
                SchedulerPredictionType.Epsilon: "v2-inference.yaml",
                SchedulerPredictionType.VPrediction: "v2-inference-v.yaml",
            },
            ModelVariantType.Inpaint: {
                SchedulerPredictionType.Epsilon: "v2-inpainting-inference.yaml",
                SchedulerPredictionType.VPrediction: "v2-inpainting-inference-v.yaml",
            },
        },
        BaseModelType.StableDiffusionXL: {
            ModelVariantType.Normal: "sd_xl_base.yaml",
        },
        BaseModelType.StableDiffusionXLRefiner: {
            ModelVariantType.Normal: "sd_xl_refiner.yaml",
        },
    }

    def __init__(
        self,
        config: Optional[InvokeAIAppConfig] = None,
        queue: Optional[DownloadQueueServiceBase] = None,
        store: Optional[ModelRecordServiceBase] = None,
        event_bus: Optional["EventServiceBase"] = None,
        event_handlers: List[DownloadEventHandler] = [],
    ):  # noqa D107 - use base class docstrings
        self._app_config = config or InvokeAIAppConfig.get_config()
        self._store = store or ModelRecordServiceBase.open(self._app_config)
        self._logger = InvokeAILogger.get_logger(config=self._app_config)
        self._event_bus = event_bus
        self._precision = get_precision()
        self._handlers = event_handlers
        if self._event_bus:
            self._handlers.append(self._event_bus.emit_model_event)

        self._download_queue = queue or DownloadQueueService(event_bus=event_bus, config=self._app_config)
        self._async_installs: Dict[Union[str, Path, AnyHttpUrl], Union[str, None]] = dict()
        self._installed = set()
        self._tmpdir = None

    def start(self, invoker: Any):  # Because .processor is giving circular import errors, declaring invoker an 'Any'
        """Call automatically at process start."""
        self.sync_to_config()

    @property
    def queue(self) -> DownloadQueueServiceBase:
        """Return the queue."""
        return self._download_queue

    @property
    def store(self) -> ModelConfigStore:
        """Return the storage backend used by the installer."""
        return self._store

    @property
    def config(self) -> InvokeAIAppConfig:
        """Return the app_config used by the installer."""
        return self._app_config

    def install_model(
        self,
        source: Union[str, Path, AnyHttpUrl],
        inplace: bool = True,
        priority: int = 10,
        variant: Optional[str] = None,
        subfolder: Optional[str] = None,
        probe_override: Optional[Dict[str, Any]] = None,
        metadata: Optional[ModelSourceMetadata] = None,
        access_token: Optional[str] = None,
    ) -> ModelInstallJob:  # noqa D102
        queue = self._download_queue
        variant = variant or ("fp16" if self._precision == "float16" else None)

        job = self._make_download_job(
            source, variant=variant, access_token=access_token, subfolder=subfolder, priority=priority
        )
        handler = (
            self._complete_registration_handler
            if inplace and Path(source).exists()
            else self._complete_installation_handler
        )
        if isinstance(job, ModelInstallJob):
            job.probe_override = probe_override
        if metadata and isinstance(job, DownloadJobWithMetadata):
            job.metadata = metadata
        job.add_event_handler(handler)

        self._async_installs[source] = None
        queue.submit_download_job(job, True)
        return job

    def register_path(
        self, model_path: Union[Path, str], overrides: Optional[Dict[str, Any]] = None
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        info: ModelProbeInfo = self._probe_model(model_path, overrides)
        return self._register(model_path, info)

    def install_path(
        self,
        model_path: Union[Path, str],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        info: ModelProbeInfo = self._probe_model(model_path, overrides)

        dest_path = self._app_config.models_path / info.base_type.value / info.model_type.value / model_path.name
        new_path = self._move_model(model_path, dest_path)
        new_hash = self.hash(new_path)
        assert new_hash == info.hash, f"{model_path}: Model hash changed during installation, possibly corrupted."
        return self._register(
            new_path,
            info,
        )

    def unregister(self, key: str):  # noqa D102
        self._store.del_model(key)

    def delete(self, key: str):  # noqa D102
        model = self._store.get_model(key)
        path = self._app_config.models_path / model.path
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()
        self.unregister(key)

    def conditionally_delete(self, key: str):  # noqa D102
        """Unregister the model. Delete its files only if they are within our models directory."""
        model = self._store.get_model(key)
        models_dir = self._app_config.models_path
        model_path = models_dir / model.path
        if model_path.is_relative_to(models_dir):
            self.delete(key)
        else:
            self.unregister(key)

    def _register(self, model_path: Path, info: ModelProbeInfo) -> str:
        key: str = FastModelHash.hash(model_path)

        model_path = model_path.absolute()
        if model_path.is_relative_to(self._app_config.models_path):
            model_path = model_path.relative_to(self._app_config.models_path)

        registration_data = dict(
            path=model_path.as_posix(),
            name=model_path.name if model_path.is_dir() else model_path.stem,
            base_model=info.base_type,
            model_type=info.model_type,
            model_format=info.format,
            hash=key,
        )
        # add 'main' specific fields
        if info.model_type == ModelType.Main:
            if info.variant_type:
                registration_data.update(variant=info.variant_type)
            if info.format == ModelFormat.Checkpoint:
                try:
                    config_file = self._legacy_configs[info.base_type][info.variant_type]
                    if isinstance(config_file, dict):  # need another tier for sd-2.x models
                        if prediction_type := info.prediction_type:
                            config_file = config_file[prediction_type]
                        else:
                            self._logger.warning(
                                f"Could not infer prediction type for {model_path.stem}. Guessing 'v_prediction' for a SD-2 768 pixel model"
                            )
                            config_file = config_file[SchedulerPredictionType.VPrediction]
                    registration_data.update(
                        config=Path(self._app_config.legacy_conf_dir, str(config_file)).as_posix(),
                    )
                except KeyError as exc:
                    raise InvalidModelException(
                        "Configuration file for this checkpoint could not be determined"
                    ) from exc
        self._store.add_model(key, registration_data)
        return key

    def _move_model(self, old_path: Path, new_path: Path) -> Path:
        if old_path == new_path:
            return old_path

        new_path.parent.mkdir(parents=True, exist_ok=True)

        # if path already exists then we jigger the name to make it unique
        counter: int = 1
        while new_path.exists():
            path = new_path.with_stem(new_path.stem + f"_{counter:02d}")
            if not path.exists():
                new_path = path
            counter += 1
        return move(old_path, new_path)

    def _probe_model(self, model_path: Union[Path, str], overrides: Optional[Dict[str, Any]] = None) -> ModelProbeInfo:
        info: ModelProbeInfo = ModelProbe.probe(Path(model_path))
        if overrides:  # used to override probe fields
            for key, value in overrides.items():
                try:
                    setattr(info, key, value)  # skip validation errors
                except Exception:
                    pass
        return info

    def _complete_installation_handler(self, job: DownloadJobBase):
        assert isinstance(job, ModelInstallJob)
        if job.status == "completed":
            self._logger.info(f"{job.source}: Download finished with status {job.status}. Installing.")
            model_id = self.install_path(job.destination, job.probe_override)
            info = self._store.get_model(model_id)
            info.source = str(job.source)
            if isinstance(job, DownloadJobWithMetadata):
                metadata: ModelSourceMetadata = job.metadata
                info.description = metadata.description or f"Imported model {info.name}"
                info.name = metadata.name or info.name
                info.author = metadata.author
                info.tags = metadata.tags
                info.license = metadata.license
                info.thumbnail_url = metadata.thumbnail_url
            self._store.update_model(model_id, info)
            self._async_installs[job.source] = model_id
            job.model_key = model_id
        elif job.status == "error":
            self._logger.warning(f"{job.source}: Model installation error: {job.error}")
        elif job.status == "cancelled":
            self._logger.warning(f"{job.source}: Model installation cancelled at caller's request.")
        jobs = self._download_queue.list_jobs()
        if self._tmpdir and len(jobs) <= 1 and job.status in ["completed", "error", "cancelled"]:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _complete_registration_handler(self, job: DownloadJobBase):
        assert isinstance(job, ModelInstallJob)
        if job.status == "completed":
            self._logger.info(f"{job.source}: Installing in place.")
            model_id = self.register_path(job.destination, job.probe_override)
            info = self._store.get_model(model_id)
            info.source = str(job.source)
            info.description = f"Imported model {info.name}"
            self._store.update_model(model_id, info)
            self._async_installs[job.source] = model_id
            job.model_key = model_id
        elif job.status == "error":
            self._logger.warning(f"{job.source}: Model installation error: {job.error}")
        elif job.status == "cancelled":
            self._logger.warning(f"{job.source}: Model installation cancelled at caller's request.")

    def sync_model_path(self, key: str, ignore_hash_change: bool = False) -> ModelConfigBase:
        """
        Move model into the location indicated by its basetype, type and name.

        Call this after updating a model's attributes in order to move
        the model's path into the location indicated by its basetype, type and
        name. Applies only to models whose paths are within the root `models_dir`
        directory.

        May raise an UnknownModelException.
        """
        model = self._store.get_model(key)
        old_path = Path(model.path)
        models_dir = self._app_config.models_path

        if not old_path.is_relative_to(models_dir):
            return model

        new_path = models_dir / model.base_model.value / model.model_type.value / model.name
        self._logger.info(f"Moving {model.name} to {new_path}.")
        new_path = self._move_model(old_path, new_path)
        model.hash = self.hash(new_path)
        model.path = new_path.relative_to(models_dir).as_posix()
        if model.hash != key:
            assert (
                ignore_hash_change
            ), f"{model.name}: Model hash changed during installation, model is possibly corrupted"
            self._logger.info(f"Model has new hash {model.hash}, but will continue to be identified by {key}")
        self._store.update_model(key, model)
        return model

    def _make_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        variant: Optional[str] = None,
        subfolder: Optional[str] = None,
        access_token: Optional[str] = None,
        priority: Optional[int] = 10,
    ) -> ModelInstallJob:
        # Clean up a common source of error. Doesn't work with Paths.
        if isinstance(source, str):
            source = source.strip()

        # In the event that we are being asked to install a path that is already on disk,
        # we simply probe and register/install it. The job does not actually do anything, but we
        # create one anyway in order to have similar behavior for local files, URLs and repo_ids.
        if Path(source).exists():  # a path that is already on disk
            destdir = source
            return ModelInstallPathJob(source=source, destination=Path(destdir), event_handlers=self._handlers)

        # choose a temporary directory inside the models directory
        models_dir = self._app_config.models_path
        self._tmpdir = self._tmpdir or tempfile.TemporaryDirectory(dir=models_dir)

        cls = ModelInstallJob
        if match := re.match(REPO_ID_WITH_OPTIONAL_SUBFOLDER_RE, str(source)):
            cls = ModelInstallRepoIDJob
            source = match.group(1)
            subfolder = match.group(2) or subfolder
            kwargs = dict(variant=variant, subfolder=subfolder)
        elif re.match(HTTP_RE, str(source)):
            cls = ModelInstallURLJob
            kwargs = {}
        else:
            raise ValueError(f"'{source}' is not recognized as a local file, directory, repo_id or URL")
        return cls(
            source=str(source),
            destination=Path(self._tmpdir.name),
            access_token=access_token,
            priority=priority,
            event_handlers=self._handlers,
            **kwargs,
        )

    def wait_for_installs(self) -> Dict[Union[str, Path, AnyHttpUrl], Optional[str]]:
        """Pause until all installation jobs have completed."""
        self._download_queue.join()
        id_map = self._async_installs
        self._async_installs = dict()
        return id_map

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:  # noqa D102
        self._cached_model_paths = set([Path(x.path) for x in self._store.all_models()])
        callback = self._scan_install if install else self._scan_register
        search = ModelSearch(on_model_found=callback)
        self._installed = set()
        search.search(scan_dir)
        return list(self._installed)

    def scan_models_directory(self):
        """
        Scan the models directory for new and missing models.

        New models will be added to the storage backend. Missing models
        will be deleted.
        """
        defunct_models = set()
        installed = set()

        with Chdir(self._app_config.models_path):
            self._logger.info("Checking for models that have been moved or deleted from disk")
            for model_config in self._store.all_models():
                path = Path(model_config.path)
                if not path.exists():
                    self._logger.info(f"{model_config.name}: path {path.as_posix()} no longer exists. Unregistering")
                    defunct_models.add(model_config.key)
            for key in defunct_models:
                self.unregister(key)

            self._logger.info(f"Scanning {self._app_config.models_path} for new models")
            for cur_base_model in BaseModelType:
                for cur_model_type in ModelType:
                    models_dir = Path(cur_base_model.value, cur_model_type.value)
                    installed.update(self.scan_directory(models_dir))
            self._logger.info(f"{len(installed)} new models registered; {len(defunct_models)} unregistered")

    def sync_to_config(self):
        """Synchronize models on disk to those in memory."""
        self.scan_models_directory()
        if autoimport := self._app_config.autoimport_dir:
            self._logger.info("Scanning autoimport directory for new models")
            self.scan_directory(self._app_config.root_path / autoimport)

    def hash(self, model_path: Union[Path, str]) -> str:  # noqa D102
        return FastModelHash.hash(model_path)

    def _scan_register(self, model: Path) -> bool:
        if model in self._cached_model_paths:
            return True
        try:
            id = self.register_path(model)
            self.sync_model_path(id)  # possibly move it to right place in `models`
            self._logger.info(f"Registered {model.name} with id {id}")
            self._installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def _scan_install(self, model: Path) -> bool:
        if model in self._cached_model_paths:
            return True
        try:
            id = self.install_path(model)
            self._logger.info(f"Installed {model} with id {id}")
            self._installed.add(id)
        except DuplicateModelException:
            pass
        return True
