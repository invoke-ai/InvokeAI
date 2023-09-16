# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Install/delete models.

Typical usage:

  from invokeai.app.services.config import InvokeAIAppConfig
  from invokeai.backend.model_manager import ModelInstall
  from invokeai.backend.model_manager.storage import ModelConfigStoreSQL
  from invokeai.backend.model_manager.download import DownloadQueue

  config = InvokeAIAppConfig.get_config()
  store = ModelConfigStoreSQL(config.db_path)
  download = DownloadQueue()
  installer = ModelInstall(store=store, config=config, download=download)

  # register config, don't move path
  id: str = installer.register_path('/path/to/model')

  # register config, and install model in `models`
  id: str = installer.install_path('/path/to/model')

  # download some remote models and install them in the background
  installer.install('stabilityai/stable-diffusion-2-1')
  installer.install('https://civitai.com/api/download/models/154208')
  installer.install('runwayml/stable-diffusion-v1-5')
  installer.install('/home/user/models/stable-diffusion-v1-5', inplace=True)

  installed_ids = installer.wait_for_installs()
  id1 = installed_ids['stabilityai/stable-diffusion-2-1']
  id2 = installed_ids['https://civitai.com/api/download/models/154208']

  # unregister, don't delete
  installer.unregister(id)

  # unregister and delete model from disk
  installer.delete_model(id)

  # scan directory recursively and install all new models found
  ids: List[str] = installer.scan_directory('/path/to/directory')

  # unregister any model whose path is no longer valid
  ids: List[str] = installer.garbage_collect()

  hash: str = installer.hash('/path/to/model')  # should be same as id above

The following exceptions may be raised:
  DuplicateModelException
  UnknownModelTypeException
"""
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import Field
from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger

from .config import BaseModelType, ModelFormat, ModelType, ModelVariantType, SchedulerPredictionType
from .download import (
    HTTP_RE,
    REPO_ID_RE,
    DownloadEventHandler,
    DownloadJobBase,
    DownloadQueue,
    DownloadQueueBase,
    ModelSourceMetadata,
)
from .download.queue import DownloadJobPath, DownloadJobRepoID, DownloadJobURL
from .hash import FastModelHash
from .probe import InvalidModelException, ModelProbe, ModelProbeInfo
from .search import ModelSearch
from .storage import DuplicateModelException, ModelConfigStore, get_config_store


class ModelInstallJob(DownloadJobBase):
    """This is a version of DownloadJobBase that has an additional slot for the model key and probe info."""

    model_key: Optional[str] = Field(
        description="After model installation, this field will hold its primary key", default=None
    )
    probe_override: Optional[Dict[str, Any]] = Field(
        description="Keys in this dict will override like-named attributes in the automatic probe info",
        default=None,
    )


class ModelInstallURLJob(DownloadJobURL, ModelInstallJob):
    """Job for installing URLs."""


class ModelInstallRepoIDJob(DownloadJobRepoID, ModelInstallJob):
    """Job for installing repo ids."""


class ModelInstallPathJob(DownloadJobPath, ModelInstallJob):
    """Job for installing local paths."""


ModelInstallEventHandler = Callable[["ModelInstallJob"], None]


class ModelInstallBase(ABC):
    """Abstract base class for InvokeAI model installation"""

    @abstractmethod
    def __init__(
        self,
        store: Optional[ModelConfigStore] = None,
        config: Optional[InvokeAIAppConfig] = None,
        logger: Optional[InvokeAILogger] = None,
        download: Optional[DownloadQueueBase] = None,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ):
        """
        Create ModelInstall object.

        :param store: Optional ModelConfigStore. If None passed,
        defaults to `configs/models.yaml`.
        :param config: Optional InvokeAIAppConfig. If None passed,
        uses the system-wide default app config.
        :param logger: Optional InvokeAILogger. If None passed,
        uses the system-wide default logger.
        :param download: Optional DownloadQueueBase object. If None passed,
        a default queue object will be created.
        :param event_handlers: List of event handlers to pass to the queue object.
        """
        pass

    @property
    @abstractmethod
    def queue(self) -> DownloadQueueBase:
        """Return the download queue used by the installer."""
        pass

    @abstractmethod
    def register_path(self, model_path: Union[Path, str], info: Optional[ModelProbeInfo] = None) -> str:
        """
        Probe and register the model at model_path.

        :param model_path: Filesystem Path to the model.
        :param info: Optional ModelProbeInfo object. If not provided, model will be probed.
        :returns id: The string ID of the registered model.
        """
        pass

    @abstractmethod
    def install_path(self, model_path: Union[Path, str], info: Optional[ModelProbeInfo] = None) -> str:
        """
        Probe, register and install the model in the models directory.

        This involves moving the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :param info: Optional ModelProbeInfo object. If not provided, model will be probed.
        :returns id: The string ID of the installed model.
        """
        pass

    @abstractmethod
    def install(
        self,
        source: Union[str, Path, AnyHttpUrl],
        inplace: bool = True,
        variant: Optional[str] = None,
        info: Optional[ModelProbeInfo] = None,
    ) -> DownloadJobBase:
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
        :param info: Optional ModelProbeInfo object. If not provided, model will be probed.
        :returns DownloadQueueBase object.

        The `inplace` flag does not affect the behavior of downloaded
        models, which are always moved into the `models` directory.
        """
        pass

    @abstractmethod
    def wait_for_installs(self) -> Dict[str, str]:
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
    def unregister(self, id: str):
        """
        Unregister the model identified by id.

        This removes the model from the registry without
        deleting the underlying model from disk.

        :param id: The string ID of the model to forget.
        :raises UnknownModelException: In the event the ID is unknown.
        """
        pass

    @abstractmethod
    def delete(self, id: str):
        """
        Unregister and delete the model identified by id.

        This removes the model from the registry and
        deletes the underlying model from disk.

        :param id: The string ID of the model to forget.
        :raises UnknownModelException: In the event the ID is unknown.
        :raises OSError: In the event the model cannot be deleted from disk.
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
    def garbage_collect(self) -> List[str]:
        """
        Unregister any models whose paths are no longer valid.

        This checks each registered model's path. Models with paths that are
        no longer found on disk will be unregistered.

        :return List[str]: Return the list of model IDs that were unregistered.
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


class ModelInstall(ModelInstallBase):
    """Model installer class handles installation from a local path."""

    _config: InvokeAIAppConfig
    _logger: InvokeAILogger
    _store: ModelConfigStore
    _download_queue: DownloadQueueBase
    _async_installs: Dict[str, str]
    _installed: Set[Path] = Field(default=set)
    _tmpdir: Optional[tempfile.TemporaryDirectory]  # used for downloads

    _legacy_configs = {
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
        store: Optional[ModelConfigStore] = None,
        config: Optional[InvokeAIAppConfig] = None,
        logger: Optional[InvokeAILogger] = None,
        download: Optional[DownloadQueueBase] = None,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ):  # noqa D107 - use base class docstrings
        self._config = config or InvokeAIAppConfig.get_config()
        self._logger = logger or InvokeAILogger.getLogger(config=self._config)
        self._store = store or get_config_store(self._config.model_conf_path)
        self._download_queue = download or DownloadQueue(config=self._config, event_handlers=event_handlers)
        self._async_installs = dict()
        self._installed = set()
        self._tmpdir = None

    @property
    def queue(self) -> DownloadQueueBase:
        """Return the queue."""
        return self._download_queue

    def register_path(
        self, model_path: Union[Path, str], overrides: Optional[Dict[str, Any]] = None
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        info: ModelProbeInfo = self._probe_model(model_path, overrides)
        return self._register(model_path, info)

    def _register(self, model_path: Path, info: ModelProbeInfo) -> str:
        key: str = FastModelHash.hash(model_path)
        registration_data = dict(
            path=model_path.as_posix(),
            name=model_path.stem,
            base_model=info.base_type,
            model_type=info.model_type,
            model_format=info.format,
        )
        # add 'main' specific fields
        if info.model_type == ModelType.Main and info.format == ModelFormat.Checkpoint:
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
                    config=Path(self._config.legacy_conf_dir, config_file).as_posix(),
                )
            except KeyError as exc:
                raise InvalidModelException("Configuration file for this checkpoint could not be determined") from exc
        self._store.add_model(key, registration_data)
        return key

    def install_path(
        self,
        model_path: Union[Path, str],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        info: ModelProbeInfo = self._probe_model(model_path, overrides)

        dest_path = self._config.models_path / info.base_type.value / info.model_type.value / model_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # if path already exists then we jigger the name to make it unique
        counter: int = 1
        while dest_path.exists():
            dest_path = dest_path.with_stem(dest_path.stem + f"_{counter:02d}")
            counter += 1

        return self._register(
            model_path.replace(dest_path),
            info,
        )

    def _probe_model(self, model_path: Union[Path, str], overrides: Optional[Dict[str, Any]] = None) -> ModelProbeInfo:
        info: ModelProbeInfo = ModelProbe.probe(model_path)
        if overrides:  # used to override probe fields
            for key, value in overrides.items():
                setattr(info, key, value)  # may generate a pydantic validation error
        return info

    def unregister(self, key: str):  # noqa D102
        self._store.del_model(key)

    def delete(self, key: str):  # noqa D102
        model = self._store.get_model(key)
        rmtree(model.path)
        self.unregister(key)

    def install(
        self,
        source: Union[str, Path, AnyHttpUrl],
        inplace: bool = True,
        variant: Optional[str] = None,
        probe_override: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
    ) -> DownloadJobBase:  # noqa D102
        queue = self._download_queue

        job = self._make_download_job(source, variant, access_token)
        handler = (
            self._complete_registration_handler
            if inplace and Path(source).exists()
            else self._complete_installation_handler
        )
        job.probe_override = probe_override
        job.add_event_handler(handler)

        self._async_installs[source] = None
        queue.submit_download_job(job, True)
        return job

    def _complete_installation_handler(self, job: DownloadJobBase):
        if job.status == "completed":
            self._logger.info(f"{job.source}: Download finished with status {job.status}. Installing.")
            model_id = self.install_path(job.destination, job.probe_override)
            info = self._store.get_model(model_id)
            info.source = str(job.source)
            metadata: ModelSourceMetadata = job.metadata
            info.description = metadata.description or f"Imported model {info.name}"
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

    def _make_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        variant: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> DownloadJobBase:
        # In the event that we are being asked to install a path that is already on disk,
        # we simply probe and register/install it. The job does not actually do anything, but we
        # create one anyway in order to have similar behavior for local files, URLs and repo_ids.
        if Path(source).exists():  # a path that is already on disk
            source = Path(source)
            destdir = source
            return ModelInstallPathJob(source=source, destination=Path(destdir))

        # choose a temporary directory inside the models directory
        models_dir = self._config.models_path
        self._tmpdir = self._tmpdir or tempfile.TemporaryDirectory(dir=models_dir)

        if re.match(REPO_ID_RE, str(source)):
            cls = ModelInstallRepoIDJob
            kwargs = dict(variant=variant)
        elif re.match(HTTP_RE, str(source)):
            cls = ModelInstallURLJob
            kwargs = {}
        else:
            raise NotImplementedError(f"Don't know what to do with this type of source: {source}")
        return cls(source=source, destination=Path(self._tmpdir.name), access_token=access_token, **kwargs)

    def wait_for_installs(self) -> Dict[str, str]:  # noqa D102
        self._download_queue.join()
        id_map = self._async_installs
        self._async_installs = dict()
        return id_map

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:  # noqa D102
        callback = self._scan_install if install else self._scan_register
        search = ModelSearch(on_model_found=callback)
        self._installed = set()
        search.search(scan_dir)
        return list(self._installed)

    def garbage_collect(self) -> List[str]:  # noqa D102
        unregistered = list()
        for model in self._store.all_models():
            path = Path(model.path)
            if not path.exists():
                self._store.del_model(model.key)
                unregistered.append(model.key)
        return unregistered

    def hash(self, model_path: Union[Path, str]) -> str:  # noqa D102
        return FastModelHash.hash(model_path)

    # the following two methods are callbacks to the ModelSearch object
    def _scan_register(self, model: Path) -> bool:
        try:
            id = self.register_path(model)
            self._logger.info(f"Registered {model} with id {id}")
            self._installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def _scan_install(self, model: Path) -> bool:
        try:
            id = self.install_path(model)
            self._logger.info(f"Installed {model} with id {id}")
            self._installed.add(id)
        except DuplicateModelException:
            pass
        return True
