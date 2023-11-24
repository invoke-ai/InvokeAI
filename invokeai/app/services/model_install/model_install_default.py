"""Model installation class."""

import threading
from hashlib import sha256
from pathlib import Path
from queue import Queue
from random import randbytes
from shutil import move, rmtree
from typing import Any, Dict, List, Optional, Union

from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    DuplicateModelException,
    InvalidModelConfigException,
)
from invokeai.backend.model_manager.hash import FastModelHash
from invokeai.backend.model_manager.probe import ModelProbe
from invokeai.backend.util.logging import InvokeAILogger

from .model_install_base import InstallStatus, ModelInstallJob, ModelInstallServiceBase

# marker that the queue is done and that thread should exit
STOP_JOB = ModelInstallJob(source="stop")


class ModelInstallService(ModelInstallServiceBase):
    """class for InvokeAI model installation."""

    _app_config: InvokeAIAppConfig
    _record_store: ModelRecordServiceBase
    _event_bus: Optional[EventServiceBase] = None
    _install_queue: Queue[ModelInstallJob]
    _install_jobs: Dict[Union[str, Path, AnyHttpUrl], ModelInstallJob]
    _logger: InvokeAILogger

    def __init__(self,
                 app_config: InvokeAIAppConfig,
                 record_store: ModelRecordServiceBase,
                 event_bus: Optional[EventServiceBase] = None
                 ):
        """
        Initialize the installer object.

        :param app_config: InvokeAIAppConfig object
        :param record_store: Previously-opened ModelRecordService database
        :param event_bus: Optional EventService object
        """
        self._app_config = app_config
        self._record_store = record_store
        self._install_queue = Queue()
        self._event_bus = event_bus
        self._logger = InvokeAILogger.get_logger(name=self.__class__.__name__)
        self._start_installer_thread()

    @property
    def app_config(self) -> InvokeAIAppConfig:  # noqa D102
        return self._app_config

    @property
    def record_store(self) -> ModelRecordServiceBase:   # noqa D102
        return self._record_store

    def _start_installer_thread(self) -> None:
        threading.Thread(target=self._install_next_item, daemon=True).start()

    def _install_next_item(self) -> None:
        done = False
        while not done:
            job = self._install_queue.get()
            if job == STOP_JOB:
                done = True
            elif job.status == InstallStatus.WAITING:
                assert job.local_path is not None
                try:
                    self._signal_job_running(job)
                    self.register_path(job.local_path)
                    self._signal_job_completed(job)
                except (OSError, DuplicateModelException, InvalidModelConfigException) as excp:
                    self._signal_job_errored(job, excp)

    def _signal_job_running(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.RUNNING
        if self._event_bus:
            self._event_bus.emit_model_install_started(str(job.source))

    def _signal_job_completed(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.COMPLETED
        if self._event_bus:
            assert job.local_path is not None
            self._event_bus.emit_model_install_completed(str(job.source), job.local_path.as_posix())

    def _signal_job_errored(self, job: ModelInstallJob, excp: Exception) -> None:
        job.set_error(excp)
        if self._event_bus:
            self._event_bus.emit_model_install_error(str(job.source), job.error_type, job.error)

    def register_path(
            self,
            model_path: Union[Path, str],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> str:    # noqa D102
        model_path = Path(model_path)
        metadata = metadata or {}
        if metadata.get('source') is None:
            metadata['source'] = model_path.as_posix()
        return self._register(model_path, metadata)

    def install_path(
            self,
            model_path: Union[Path, str],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> str:    # noqa D102
        model_path = Path(model_path)
        metadata = metadata or {}
        if metadata.get('source') is None:
            metadata['source'] = model_path.as_posix()

        info: AnyModelConfig = self._probe_model(Path(model_path), metadata)

        old_hash = info.original_hash
        dest_path = self.app_config.models_path / info.base.value / info.type.value / model_path.name
        new_path = self._move_model(model_path, dest_path)
        new_hash = FastModelHash.hash(new_path)
        assert new_hash == old_hash, f"{model_path}: Model hash changed during installation, possibly corrupted."

        return self._register(
            new_path,
            metadata,
            info,
        )

    def import_model(
            self,
            source: Union[str, Path, AnyHttpUrl],
            inplace: bool = True,
            variant: Optional[str] = None,
            subfolder: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
            access_token: Optional[str] = None,
    ) -> ModelInstallJob:      # noqa D102
        raise NotImplementedError

    def wait_for_installs(self) -> Dict[Union[str, Path, AnyHttpUrl], ModelInstallJob]:      # noqa D102
        self._install_queue.join()
        return self._install_jobs

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:      # noqa D102
        raise NotImplementedError

    def sync_to_config(self) -> None:      # noqa D102
        raise NotImplementedError

    def unregister(self, key: str) -> None:      # noqa D102
        self.record_store.del_model(key)

    def delete(self, key: str) -> None:      # noqa D102
        model = self.record_store.get_model(key)
        path = self.app_config.models_path / model.path
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()
        self.unregister(key)

    def conditionally_delete(self, key: str) -> None:  # noqa D102
        """Unregister the model. Delete its files only if they are within our models directory."""
        model = self.record_store.get_model(key)
        models_dir = self.app_config.models_path
        model_path = models_dir / model.path
        if model_path.is_relative_to(models_dir):
            self.delete(key)
        else:
            self.unregister(key)

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

    def _probe_model(self, model_path: Path, metadata: Optional[Dict[str, Any]] = None) -> AnyModelConfig:
        info: AnyModelConfig = ModelProbe.probe(Path(model_path))
        if metadata:  # used to override probe fields
            for key, value in metadata.items():
                setattr(info, key, value)
        return info

    def _create_key(self) -> str:
        return sha256(randbytes(100)).hexdigest()[0:32]

    def _register(self,
                  model_path: Path,
                  metadata: Optional[Dict[str, Any]] = None,
                  info: Optional[AnyModelConfig] = None) -> str:

        info = info or ModelProbe.probe(model_path, metadata)
        key = self._create_key()

        model_path = model_path.absolute()
        if model_path.is_relative_to(self.app_config.models_path):
            model_path = model_path.relative_to(self.app_config.models_path)

        info.path = model_path.as_posix()

        # add 'main' specific fields
        if hasattr(info, 'config'):
            # make config relative to our root
            info.config = self.app_config.legacy_conf_dir / info.config
        self.record_store.add_model(key, info)
        return key
