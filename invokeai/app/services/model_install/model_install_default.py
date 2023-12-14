"""Model installation class."""

import threading
from hashlib import sha256
from logging import Logger
from pathlib import Path
from queue import Queue
from random import randbytes
from shutil import copyfile, copytree, move, rmtree
from typing import Any, Dict, List, Optional, Set, Union

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.model_records import DuplicateModelException, ModelRecordServiceBase, UnknownModelException
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    InvalidModelConfigException,
    ModelType,
)
from invokeai.backend.model_manager.hash import FastModelHash
from invokeai.backend.model_manager.probe import ModelProbe
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.util import Chdir, InvokeAILogger

from .model_install_base import (
    InstallStatus,
    LocalModelSource,
    ModelInstallJob,
    ModelInstallServiceBase,
    ModelSource,
)

# marker that the queue is done and that thread should exit
STOP_JOB = ModelInstallJob(
    source=LocalModelSource(path="stop"),
    local_path=Path("/dev/null"),
)


class ModelInstallService(ModelInstallServiceBase):
    """class for InvokeAI model installation."""

    _app_config: InvokeAIAppConfig
    _record_store: ModelRecordServiceBase
    _event_bus: Optional[EventServiceBase] = None
    _install_queue: Queue[ModelInstallJob]
    _install_jobs: List[ModelInstallJob]
    _logger: Logger
    _cached_model_paths: Set[Path]
    _models_installed: Set[str]

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
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
        self._start_installer_thread()
        self.sync_to_config()

    def stop(self, *args: Any, **kwarg: Any) -> None:
        """Stop the installer thread; after this the object can be deleted and garbage collected."""
        self._install_queue.put(STOP_JOB)

    def _start_installer_thread(self) -> None:
        threading.Thread(target=self._install_next_item, daemon=True).start()

    def _install_next_item(self) -> None:
        done = False
        while not done:
            job = self._install_queue.get()
            if job == STOP_JOB:
                done = True
                continue

            assert job.local_path is not None
            try:
                self._signal_job_running(job)
                if job.inplace:
                    key = self.register_path(job.local_path, job.config_in)
                else:
                    key = self.install_path(job.local_path, job.config_in)
                job.config_out = self.record_store.get_model(key)
                self._signal_job_completed(job)

            except (OSError, DuplicateModelException, InvalidModelConfigException) as excp:
                self._signal_job_errored(job, excp)
            finally:
                self._install_queue.task_done()
        self._logger.info("Install thread exiting")

    def _signal_job_running(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.RUNNING
        self._logger.info(f"{job.source}: model installation started")
        if self._event_bus:
            self._event_bus.emit_model_install_started(str(job.source))

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
            job = ModelInstallJob(
                source=source,
                config_in=config,
                local_path=Path(source.path),
            )
            self._install_jobs.append(job)
            self._install_queue.put(job)
            return job

        else:  # here is where we'd download a URL or repo_id. Implementation pending download queue.
            raise UnknownModelException("File or directory not found")

    def list_jobs(self) -> List[ModelInstallJob]:  # noqa D102
        return self._install_jobs

    def get_job(self, source: ModelSource) -> List[ModelInstallJob]:  # noqa D102
        return [x for x in self._install_jobs if x.source == source]

    def wait_for_installs(self) -> List[ModelInstallJob]:  # noqa D102
        self._install_queue.join()
        return self._install_jobs

    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""
        unfinished_jobs = [
            x for x in self._install_jobs if x.status not in [InstallStatus.COMPLETED, InstallStatus.ERROR]
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
