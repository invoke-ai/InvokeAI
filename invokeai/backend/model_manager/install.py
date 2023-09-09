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
  id: str = installer.register_model('/path/to/model')

  # register config, and install model in `models`
  id: str = installer.install_model('/path/to/model')

  # download some remote models and install them in the background
  installer.download('stabilityai/stable-diffusion-2-1')
  installer.download('https://civitai.com/api/download/models/154208')
  installer.download('runwayml/stable-diffusion-v1-5')

  installed_ids = installer.wait_for_downloads()
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
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Optional, List, Union, Dict
from pydantic.networks import AnyHttpUrl
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger
from .search import ModelSearch
from .storage import ModelConfigStore, ModelConfigStoreYAML, DuplicateModelException
from .download import DownloadQueueBase, DownloadQueue, DownloadJobBase
from .hash import FastModelHash
from .probe import ModelProbe, ModelProbeInfo, InvalidModelException
from .config import (
    ModelType,
    BaseModelType,
    ModelVariantType,
    ModelFormat,
    SchedulerPredictionType,
)


class ModelInstallBase(ABC):
    """Abstract base class for InvokeAI model installation"""

    @abstractmethod
    def __init__(
        self,
        store: Optional[ModelConfigStore] = None,
        config: Optional[InvokeAIAppConfig] = None,
        logger: Optional[InvokeAILogger] = None,
    ):
        """
        Create ModelInstall object.

        :param store: Optional ModelConfigStore. If None passed,
        defaults to `configs/models.yaml`.
        :param config: Optional InvokeAIAppConfig. If None passed,
        uses the system-wide default app config.
        :param logger: Optional InvokeAILogger. If None passed,
        uses the system-wide default logger.
        """
        pass

    @abstractmethod
    def register(self, model_path: Union[Path, str]) -> str:
        """
        Probe and register the model at model_path.

        :param model_path: Filesystem Path to the model.
        :returns id: The string ID of the registered model.
        """
        pass

    @abstractmethod
    def install(self, model_path: Union[Path, str]) -> str:
        """
        Probe, register and install the model in the models directory.

        This involves moving the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :returns id: The string ID of the installed model.
        """
        pass

    @abstractmethod
    def download(self, source: Union[str, AnyHttpUrl]) -> DownloadJobBase:
        """
        Download and install the model located at remote site.

        This will download the model located at `source`,
        probe it, and install it into the models directory.
        This call is executed asynchronously in a separate
        thread, and the returned object is a
        invokeai.backend.model_manager.download.DownloadJobBase
        object which can be interrogated to get the status of
        the download and install process. Call our `wait_for_downloads()`
        method to wait for all downloads to complete.

        :param source: Either a URL or a HuggingFace repo_id.
        :returns queue: DownloadQueueBase object.
        """
        pass

    @abstractmethod
    def wait_for_downloads(self) -> Dict[str, str]:
        """
        Wait for all pending downloads to complete.

        This will block until all pending downloads have
        completed, been cancelled, or errored out. It will
        block indefinitely if one or more jobs are in the
        paused state.

        It will return a dict that maps the source model
        URL or repo_id to the ID of the installed model.
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
    ):  # noqa D107 - use base class docstrings
        self._config = config or InvokeAIAppConfig.get_config()
        self._logger = logger or InvokeAILogger.getLogger()
        self._store = store or ModelConfigStoreYAML(self._config.model_conf_path)
        self._download_queue = download or DownloadQueue()
        self._async_installs = dict()
        self._tmpdir = None

    def register(self, model_path: Union[Path, str]) -> str:  # noqa D102
        model_path = Path(model_path)
        info: ModelProbeInfo = ModelProbe.probe(model_path)
        return self._register(model_path, info)

    def _register(self, model_path: Path, info: ModelProbeInfo) -> str:
        id: str = FastModelHash.hash(model_path)
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
            except KeyError as exc:
                raise InvalidModelException("Configuration file for this checkpoint could not be determined") from exc
            registration_data.update(
                config=Path(self._config.legacy_conf_dir, config_file).as_posix(),
            )
        self._store.add_model(id, registration_data)
        return id

    def install(self, model_path: Union[Path, str]) -> str:  # noqa D102
        model_path = Path(model_path)
        info: ModelProbeInfo = ModelProbe.probe(model_path)
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

    def unregister(self, id: str):  # noqa D102
        self._store.del_model(id)

    def delete(self, id: str):  # noqa D102
        model = self._store.get_model(id)
        rmtree(model.path)
        self.unregister(id)

    def download(self, source: Union[str, AnyHttpUrl]) -> DownloadJobBase:  # noqa D102
        # choose a temporary directory inside the models directory
        models_dir = self._config.models_path
        queue = self._download_queue
        self._async_installs[source] = None

        def complete_installation(job: DownloadJobBase):
            self._logger.info(f"{job.source}: {job.status} filename={job.destination}({job.bytes}/{job.total_bytes})")
            if job.status == "completed":
                id = self.install(job.destination)
                info = self._store.get_model(id)
                info.description = f"Downloaded model {info.name}"
                info.source_url = str(job.source)
                self._store.update_model(id, info)
                self._async_installs[job.source] = id
            jobs = queue.list_jobs()
            if len(jobs) <= 1 and job.status in ["completed", "error", "cancelled"]:
                self._tmpdir = None

        # note - this is probably not going to work. The tmpdir
        # will be deleted before the job actually runs.
        # Better to do the cleanup in the callback
        self._tmpdir = self._tmpdir or tempfile.TemporaryDirectory(dir=models_dir)
        return queue.create_download_job(
            source=source,
            destdir=self._tmpdir.name,
            event_handlers=[complete_installation]
        )

    def wait_for_downloads(self) -> Dict[str, str]:  # noqa D102
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
                self._store.del_model(model.id)
                unregistered.append(model.id)
        return unregistered

    def hash(self, model_path: Union[Path, str]) -> str:  # noqa D102
        return FastModelHash.hash(model_path)

    # the following two methods are callbacks to the ModelSearch object
    def _scan_register(self, model: Path) -> bool:
        try:
            id = self.register(model)
            self._logger.info(f"Registered {model} with id {id}")
            self._installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def _scan_install(self, model: Path) -> bool:
        try:
            id = self.install(model)
            self._logger.info(f"Installed {model} with id {id}")
            self._installed.add(id)
        except DuplicateModelException:
            pass
        return True
