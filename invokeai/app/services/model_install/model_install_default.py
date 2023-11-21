"""Model installation class."""

import threading

from pathlib import Path
from typing import Dict, Optional, Union, List
from queue import Queue
from pydantic.networks import AnyHttpUrl

from .model_install_base import InstallStatus, ModelInstallJob, ModelInstallServiceBase

from invokeai.backend.model_management.model_probe import ModelProbeInfo, ModelProbe
from invokeai.backend.model_manager.config import InvalidModelConfigException, DuplicateModelException

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.app.services.events import EventServiceBase
from invokeai.backend.util.logging import InvokeAILogger

# marker that the queue is done and that thread should exit
STOP_JOB = ModelInstallJob(source="stop")


class ModelInstallService(ModelInstallServiceBase):
    """class for InvokeAI model installation."""

    config: InvokeAIAppConfig
    store: ModelRecordServiceBase
    _event_bus: Optional[EventServiceBase] = None
    _install_queue: Queue

    def __init__(self,
                 config: InvokeAIAppConfig,
                 store: ModelRecordServiceBase,
                 install_queue: Optional[Queue] = None,
                 event_bus: Optional[EventServiceBase] = None
                 ):
        self.config = config
        self.store = store
        self._install_queue = install_queue or Queue()
        self._event_bus = event_bus
        self._start_installer_thread()

    def _start_installer_thread(self):
        threading.Thread(target=self._install_next_item, daemon=True).start()

    def _install_next_item(self):
        done = False
        while not done:
            job = self._install_queue.get()
            if job == STOP_JOB:
                done = True
            elif job.status == InstallStatus.WAITING:
                try:
                    self._signal_job_running(job)
                    self.register_path(job.path)
                    self._signal_job_completed(job)
                except (OSError, DuplicateModelException, InvalidModelConfigException) as e:
                    self._signal_job_errored(job, e)

    def _signal_job_running(self, job: ModelInstallJob):
        job.status = InstallStatus.RUNNING
        if self._event_bus:
            self._event_bus.emit_model_install_started(job.source)

    def _signal_job_completed(self, job: ModelInstallJob):
        job.status = InstallStatus.COMPLETED
        if self._event_bus:
            self._event_bus.emit_model_install_completed(job.source, job.dest)

    def _signal_job_errored(self, job: ModelInstallJob, e: Exception):
        job.set_error(e)
        if self._event_bus:
            self._event_bus.emit_model_install_error(job.source, job.error_type, job.error)

    def register_path(
            self,
            model_path: Union[Path, str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        raise NotImplementedError

    def install_path(
            self,
            model_path: Union[Path, str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        raise NotImplementedError

    def install_model(
            self,
            source: Union[str, Path, AnyHttpUrl],
            inplace: bool = True,
            name: Optional[str] = None,
            description: Optional[str] = None,
            variant: Optional[str] = None,
            subfolder: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
            access_token: Optional[str] = None,
    ) -> ModelInstallJob:
        raise NotImplementedError

    def wait_for_installs(self) -> Dict[Union[str, Path, AnyHttpUrl], Optional[str]]:
        raise NotImplementedError

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:
        raise NotImplementedError

    def sync_to_config(self):
        raise NotImplementedError

    def hash(self, model_path: Union[Path, str]) -> str:
        raise NotImplementedError

    # The following are internal methods
    def _create_name(self, model_path: Union[Path, str]) -> str:
        model_path = Path(model_path)
        if model_path.suffix in {'.safetensors', '.bin', '.pt', '.ckpt'}:
            return model_path.stem
        else:
            return model_path.name

    def _create_description(self, model_path: Union[Path, str], info: Optional[ModelProbeInfo] = None) -> str:
        info = info or ModelProbe.probe(Path(model_path))
        name: str = self._create_name(model_path)
        return f"a {info.model_type} model {name} based on {info.base_type}"

    def _create_id(self, model_path: Union[Path, str], name: Optional[str] = None) -> str:
        name: str = name or self._create_name(model_path)
        raise NotImplementedError
