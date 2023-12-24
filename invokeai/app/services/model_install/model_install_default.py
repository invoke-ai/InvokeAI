"""Model installation class."""

import re
import threading
from hashlib import sha256
from logging import Logger
from pathlib import Path
from queue import Empty, Queue
from random import randbytes
from shutil import copyfile, copytree, move, rmtree
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadJob, DownloadJobStatus, DownloadQueueServiceBase
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.model_records import DuplicateModelException, ModelRecordServiceBase
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    DiffusersVariant,
    InvalidModelConfigException,
    ModelType,
)
from invokeai.backend.model_manager.hash import FastModelHash
from invokeai.backend.model_manager.metadata import (
    AnyModelRepoMetadata,
    CivitaiMetadata,
    CivitaiMetadataFetch,
    HuggingFaceMetadata,
    HuggingFaceMetadataFetch,
    ModelMetadataStore,
)
from invokeai.backend.model_manager.probe import ModelProbe
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.util import Chdir, InvokeAILogger

from .model_install_base import (
    HFModelSource,
    InstallStatus,
    LocalModelSource,
    ModelInstallJob,
    ModelInstallServiceBase,
    ModelSource,
    URLModelSource,
)

TMPDIR_PREFIX = "INVOKEAI_INSTALLER_TMP_"


class ModelInstallService(ModelInstallServiceBase):
    """class for InvokeAI model installation."""

    _app_config: InvokeAIAppConfig
    _record_store: ModelRecordServiceBase
    _event_bus: Optional[EventServiceBase] = None
    _install_queue: Queue[ModelInstallJob]
    _install_jobs: List[ModelInstallJob]
    _running: bool
    _logger: Logger
    _lock: threading.Lock
    _stop_event: threading.Event
    _cached_model_paths: Set[Path]
    _download_queue: DownloadQueueServiceBase
    _download_exited_event: threading.Event
    _models_installed: Set[str]
    _metadata_store: ModelMetadataStore
    _metadata_cache: Dict[ModelSource, Optional[AnyModelRepoMetadata]]
    _download_cache: Dict[AnyHttpUrl, ModelInstallJob]

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        metadata_store: ModelMetadataStore,
        event_bus: Optional[EventServiceBase] = None,
    ):
        """
        Initialize the installer object.

        :param app_config: InvokeAIAppConfig object
        :param record_store: Previously-opened ModelRecordService database
        :param event_bus: Optional EventService object
        """
        self._app_config = app_config
        self._record_store = record_store
        self._event_bus = event_bus
        self._logger = InvokeAILogger.get_logger(name=self.__class__.__name__)
        self._install_jobs = []
        self._install_queue = Queue()
        self._cached_model_paths = set()
        self._models_installed = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._download_exited_event = threading.Event()
        self._download_queue = download_queue
        self._metadata_cache = {}
        self._download_cache = {}
        self._metadata_store = metadata_store
        self._running = False

    @property
    def app_config(self) -> InvokeAIAppConfig:  # noqa D102
        return self._app_config

    @property
    def record_store(self) -> ModelRecordServiceBase:  # noqa D102
        return self._record_store

    @property
    def event_bus(self) -> Optional[EventServiceBase]:  # noqa D102
        return self._event_bus

    def start(self, *args: Any, **kwarg: Any) -> None:
        """Start the installer thread."""
        with self._lock:
            if self._running:
                raise Exception("Attempt to start the installer service twice")
            self._start_installer_thread()
            self.sync_to_config()

    def stop(self, *args: Any, **kwarg: Any) -> None:
        """Stop the installer thread; after this the object can be deleted and garbage collected."""
        with self._lock:
            if not self._running:
                raise Exception("Attempt to stop the install service before it was started")
            self._stop_event.set()
            with self._install_queue.mutex:
                self._install_queue.queue.clear()  # get rid of pending jobs
            active_jobs = [x for x in self.list_jobs() if x.status == InstallStatus.RUNNING]
            if active_jobs:
                self._logger.warning("Waiting for active install job to complete")
            self.wait_for_installs()
            self._download_cache.clear()
            self._metadata_cache.clear()
            self._running = False

    def _start_installer_thread(self) -> None:
        threading.Thread(target=self._install_next_item, daemon=True).start()
        self._running = True

    def _install_next_item(self) -> None:
        done = False
        while not done:
            if self._stop_event.is_set():
                done = True
                continue
            try:
                job = self._install_queue.get(timeout=1)
            except Empty:
                continue

            assert job.local_path is not None
            try:
                if job.cancelled:
                    self._signal_job_cancelled(job)
                else:
                    self._signal_job_running(job)
                    if job.inplace:
                        key = self.register_path(job.local_path, job.config_in)
                    else:
                        key = self.install_path(job.local_path, job.config_in)
                    job.config_out = self.record_store.get_model(key)

                    # enter the metadata, if there is any
                    if metadata := self._metadata_cache.get(job.source):
                        self._metadata_store.add_metadata(key, metadata)
                    self._signal_job_completed(job)

            except (OSError, DuplicateModelException, InvalidModelConfigException) as excp:
                self._signal_job_errored(job, excp)

            finally:
                # if this is an install of a remote file, then clean up the temporary directory
                if not isinstance(job.source, LocalModelSource):
                    rmtree(job.local_path.parent)
                self._metadata_cache.pop(job.source, None)
                self._install_queue.task_done()

        self._logger.info("Install thread exiting")

    def _signal_job_running(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.RUNNING
        self._logger.info(f"{job.source}: model installation started")
        if self._event_bus:
            self._event_bus.emit_model_install_running(str(job.source))

    def _signal_job_progress(self, job: ModelInstallJob) -> None:
        if self._event_bus:
            parts = [
                {"url": str(x.source), "path": str(x.download_path), "bytes": x.bytes, "total_bytes": x.total_bytes}
                for x in job.download_parts
            ]
            self._event_bus.emit_model_install_downloading(str(job.source), parts=parts)

    def _signal_job_completed(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.COMPLETED
        assert job.config_out
        self._logger.info(
            f"{job.source}: model installation completed. {job.local_path} registered key {job.config_out.key}"
        )
        if self._event_bus:
            assert job.local_path is not None
            assert job.config_out is not None
            key = job.config_out.key
            self._event_bus.emit_model_install_completed(str(job.source), key)

    def _signal_job_errored(self, job: ModelInstallJob, excp: Exception) -> None:
        job.set_error(excp)
        self._logger.info(f"{job.source}: model installation encountered an exception: {job.error_type}")
        if self._event_bus:
            error_type = job.error_type
            error = job.error
            assert error_type is not None
            assert error is not None
            self._event_bus.emit_model_install_error(str(job.source), error_type, error)

    def _signal_job_cancelled(self, job: ModelInstallJob) -> None:
        self._logger.info(f"{job.source}: model installation was cancelled")
        if self._event_bus:
            self._event_bus.emit_model_install_cancelled(str(job.source))

    def register_path(
        self,
        model_path: Union[Path, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        config = config or {}
        if config.get("source") is None:
            config["source"] = model_path.resolve().as_posix()
        return self._register(model_path, config)

    def install_path(
        self,
        model_path: Union[Path, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        config = config or {}
        if config.get("source") is None:
            config["source"] = model_path.resolve().as_posix()

        info: AnyModelConfig = self._probe_model(Path(model_path), config)
        old_hash = info.original_hash
        dest_path = self.app_config.models_path / info.base.value / info.type.value / model_path.name
        new_path = self._copy_model(model_path, dest_path)
        new_hash = FastModelHash.hash(new_path)
        assert new_hash == old_hash, f"{model_path}: Model hash changed during installation, possibly corrupted."

        return self._register(
            new_path,
            config,
            info,
        )

    def import_model(
        self,
        source: ModelSource,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelInstallJob:  # noqa D102
        if not config:
            config = {}

        # Installing a local path
        if isinstance(source, LocalModelSource) and Path(source.path).exists():  # a path that is already on disk
            install_job = ModelInstallJob(
                source=source,
                config_in=config,
                local_path=Path(source.path),
            )
            self._install_queue.put(install_job)

        else:  # a remote model
            install_job = ModelInstallJob(
                source=source,
                config_in=config,
                local_path=Path(mkdtemp(dir=self._app_config.models_path, prefix=TMPDIR_PREFIX)),
            )

            metadata = self._get_metadata(source)  #  may return None
            url_and_paths = self._get_download_urls(source, metadata)
            self._metadata_cache[source] = metadata  #  save for installation time

            for url, path in url_and_paths:
                assert hasattr(source, "access_token")
                dest = install_job.local_path / path.parent
                dest.mkdir(parents=True, exist_ok=True)
                download_job = DownloadJob(
                    source=url,
                    dest=dest,
                    access_token=source.access_token,
                )
                self._download_cache[download_job.source] = install_job  # matches a download job to an install job
                install_job.download_parts.add(download_job)

                self._download_queue.submit_download_job(
                    download_job,
                    on_start=self._download_started_callback,
                    on_progress=self._download_progress_callback,
                    on_complete=self._download_complete_callback,
                    on_error=self._download_error_callback,
                    on_cancelled=self._download_cancelled_callback,
                )
        self._install_jobs.append(install_job)
        return install_job

    def list_jobs(self) -> List[ModelInstallJob]:  # noqa D102
        return self._install_jobs

    def get_job(self, source: ModelSource) -> List[ModelInstallJob]:  # noqa D102
        return [x for x in self._install_jobs if x.source == source]

    def wait_for_installs(self) -> List[ModelInstallJob]:  # noqa D102
        while len(self._download_cache) > 0:
            self._download_exited_event.wait()
            self._download_exited_event.clear()
        self._install_queue.join()
        return self._install_jobs

    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""
        unfinished_jobs = [
            x
            for x in self._install_jobs
            if x.status not in [InstallStatus.COMPLETED, InstallStatus.ERROR, InstallStatus.CANCELLED]
        ]
        self._install_jobs = unfinished_jobs

    def sync_to_config(self) -> None:
        """Synchronize models on disk to those in the config record store database."""
        self._scan_models_directory()
        if autoimport := self._app_config.autoimport_dir:
            self._logger.info("Scanning autoimport directory for new models")
            installed = self.scan_directory(self._app_config.root_path / autoimport)
            self._logger.info(f"{len(installed)} new models registered")
        self._logger.info("Model installer (re)initialized")

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:  # noqa D102
        self._cached_model_paths = {Path(x.path) for x in self.record_store.all_models()}
        callback = self._scan_install if install else self._scan_register
        search = ModelSearch(on_model_found=callback)
        self._models_installed: Set[str] = set()
        search.search(scan_dir)
        return list(self._models_installed)

    def _scan_models_directory(self) -> None:
        """
        Scan the models directory for new and missing models.

        New models will be added to the storage backend. Missing models
        will be deleted.
        """
        defunct_models = set()
        installed = set()

        with Chdir(self._app_config.models_path):
            self._logger.info("Checking for models that have been moved or deleted from disk")
            for model_config in self.record_store.all_models():
                path = Path(model_config.path)
                if not path.exists():
                    self._logger.info(f"{model_config.name}: path {path.as_posix()} no longer exists. Unregistering")
                    defunct_models.add(model_config.key)
            for key in defunct_models:
                self.unregister(key)

            self._logger.info(f"Scanning {self._app_config.models_path} for new and orphaned models")
            for cur_base_model in BaseModelType:
                for cur_model_type in ModelType:
                    models_dir = Path(cur_base_model.value, cur_model_type.value)
                    installed.update(self.scan_directory(models_dir))
            self._logger.info(f"{len(installed)} new models registered; {len(defunct_models)} unregistered")

    def _sync_model_path(self, key: str, ignore_hash_change: bool = False) -> AnyModelConfig:
        """
        Move model into the location indicated by its basetype, type and name.

        Call this after updating a model's attributes in order to move
        the model's path into the location indicated by its basetype, type and
        name. Applies only to models whose paths are within the root `models_dir`
        directory.

        May raise an UnknownModelException.
        """
        model = self.record_store.get_model(key)
        old_path = Path(model.path)
        models_dir = self.app_config.models_path

        if not old_path.is_relative_to(models_dir):
            return model

        new_path = models_dir / model.base.value / model.type.value / model.name
        self._logger.info(f"Moving {model.name} to {new_path}.")
        new_path = self._move_model(old_path, new_path)
        new_hash = FastModelHash.hash(new_path)
        model.path = new_path.relative_to(models_dir).as_posix()
        if model.current_hash != new_hash:
            assert (
                ignore_hash_change
            ), f"{model.name}: Model hash changed during installation, model is possibly corrupted"
            model.current_hash = new_hash
            self._logger.info(f"Model has new hash {model.current_hash}, but will continue to be identified by {key}")
        self.record_store.update_model(key, model)
        return model

    def _scan_register(self, model: Path) -> bool:
        if model in self._cached_model_paths:
            return True
        try:
            id = self.register_path(model)
            self._sync_model_path(id)  # possibly move it to right place in `models`
            self._logger.info(f"Registered {model.name} with id {id}")
            self._models_installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def _scan_install(self, model: Path) -> bool:
        if model in self._cached_model_paths:
            return True
        try:
            id = self.install_path(model)
            self._logger.info(f"Installed {model} with id {id}")
            self._models_installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def unregister(self, key: str) -> None:  # noqa D102
        self.record_store.del_model(key)

    def delete(self, key: str) -> None:  # noqa D102
        """Unregister the model. Delete its files only if they are within our models directory."""
        model = self.record_store.get_model(key)
        models_dir = self.app_config.models_path
        model_path = models_dir / model.path
        if model_path.is_relative_to(models_dir):
            self.unconditionally_delete(key)
        else:
            self.unregister(key)

    def unconditionally_delete(self, key: str) -> None:  # noqa D102
        model = self.record_store.get_model(key)
        path = self.app_config.models_path / model.path
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()
        self.unregister(key)

    def _copy_model(self, old_path: Path, new_path: Path) -> Path:
        if old_path == new_path:
            return old_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if old_path.is_dir():
            copytree(old_path, new_path)
        else:
            copyfile(old_path, new_path)
        return new_path

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
        move(old_path, new_path)
        return new_path

    def _probe_model(self, model_path: Path, config: Optional[Dict[str, Any]] = None) -> AnyModelConfig:
        info: AnyModelConfig = ModelProbe.probe(Path(model_path))
        if config:  # used to override probe fields
            for key, value in config.items():
                setattr(info, key, value)
        return info

    def _create_key(self) -> str:
        return sha256(randbytes(100)).hexdigest()[0:32]

    def _register(
        self, model_path: Path, config: Optional[Dict[str, Any]] = None, info: Optional[AnyModelConfig] = None
    ) -> str:
        info = info or ModelProbe.probe(model_path, config)
        key = self._create_key()

        model_path = model_path.absolute()
        if model_path.is_relative_to(self.app_config.models_path):
            model_path = model_path.relative_to(self.app_config.models_path)

        info.path = model_path.as_posix()

        # add 'main' specific fields
        if hasattr(info, "config"):
            # make config relative to our root
            legacy_conf = (self.app_config.root_dir / self.app_config.legacy_conf_dir / info.config).resolve()
            info.config = legacy_conf.relative_to(self.app_config.root_dir).as_posix()
        self.record_store.add_model(key, info)
        return key

    def _get_metadata(self, source: ModelSource) -> Optional[AnyModelRepoMetadata]:
        url_patterns = {
            r"https?://civitai.com/": CivitaiMetadataFetch,
            r"https?://huggingface.co/": HuggingFaceMetadataFetch,
        }

        if hasattr(source, "url"):
            for pattern, fetcher in url_patterns.items():
                if re.match(pattern, str(source.url)):
                    return fetcher().from_url(source.url)
            return None

        elif isinstance(source, HFModelSource):
            return HuggingFaceMetadataFetch().from_id(source.repo_id)

        else:
            raise NotImplementedError(f"Do not know to do with a model source of type {type(source)}")

    # Huggingface is  complicated because multiple files need to be downloaded into
    # their proper subfolders. Furthermore, depending on the variant and subfolder
    # requested, there are different sets of files to install. `HuggingFaceMetadataFetch.list_download_urls()`
    # provides the logic to do this.
    def _get_download_urls(
        self, source: ModelSource, metadata: Optional[AnyModelRepoMetadata] = None
    ) -> List[Tuple[AnyHttpUrl, Path]]:
        if isinstance(source, URLModelSource):
            if isinstance(metadata, CivitaiMetadata):
                return [(metadata.download_url, Path("."))]  # download the file in the metadata URL
            else:
                return [(source.url, Path("."))]  # download the requested URL into temporary directory

        elif isinstance(source, HFModelSource):
            assert isinstance(metadata, HuggingFaceMetadata)
            return HuggingFaceMetadataFetch().list_download_urls(
                metadata,
                variant=DiffusersVariant(source.variant) if source.variant else None,
                subfolder=Path(source.subfolder) if source.subfolder else None,
            )
        else:
            raise Exception("Don't know how to get the URL for a source of type {type(source)}")

    def _download_started_callback(self, download_job: DownloadJob) -> None:
        install_job = self._download_cache[download_job.source]
        # When the download starts, the file download name may change due to Content-disposition handling
        # We patch it up here, but only for URL downloads.
        if isinstance(install_job.source, URLModelSource):
            assert download_job.download_path
            install_job.local_path = download_job.download_path
        elif isinstance(install_job.source, HFModelSource):
            assert download_job.download_path
            # If our installer local path is the original tmp directory, then we need
            # to alter it to point to the directory containing the diffusers model.
            # This hacky method relies on recognizing the tmp directory by its prefix.
            if install_job.local_path.name.startswith(TMPDIR_PREFIX):
                partial_path = download_job.download_path.relative_to(install_job.local_path)
                diffusers_folder_name = partial_path.parts[0]
                install_job.local_path = install_job.local_path / diffusers_folder_name

    def _download_progress_callback(self, download_job: DownloadJob) -> None:
        install_job = self._download_cache[download_job.source]
        self._signal_job_progress(install_job)

    def _download_complete_callback(self, download_job: DownloadJob) -> None:
        with self._lock:
            install_job = self._download_cache[download_job.source]
            self._download_cache.pop(download_job.source, None)
            self._download_exited_event.set()  # this lets wait_for_installs() know that the active job count has changed
            # are there any more active jobs left in this task?
            if all(x.status == DownloadJobStatus.COMPLETED for x in install_job.download_parts):
                self._install_queue.put(
                    install_job
                )  #  now enqueue job for actual installation into the models directory

    def _download_error_callback(self, download_job: DownloadJob, excp: Optional[Exception] = None) -> None:
        install_job = self._download_cache[download_job.source]
        assert excp is not None
        self._download_cache.pop(download_job.source, None)
        self._download_exited_event.set()  # this lets wait_for_installs() know that the active job count has changed

        # if there are other jobs in the download queue, we cancel them
        pending_jobs = [
            x
            for x in self._download_queue.list_jobs()
            if x.status in [DownloadJobStatus.RUNNING, DownloadJobStatus.WAITING]
        ]
        sibling_jobs = [x for x in pending_jobs if self._download_cache[x.source] == install_job]
        for s in sibling_jobs:
            self._download_queue.cancel_job(s)
        install_job.cancel()
        self._signal_job_errored(install_job, excp)

    def _download_cancelled_callback(self, download_job: DownloadJob) -> None:
        pass  # have to cancel other jobs, if any, but this will lead to a loop if not careful
