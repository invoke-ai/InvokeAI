import traceback
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.model_records import ModelRecordServiceBase


class InstallStatus(str, Enum):
    """State of an install job running in the background."""

    WAITING = "waiting"      # waiting to be dequeued
    RUNNING = "running"      # being processed
    COMPLETED = "completed"  # finished running
    ERROR = "error"          # terminated with an error message


class UnknownInstallJobException(Exception):
    """Raised when the status of an unknown job is requested."""


ModelSource = Union[str, Path, AnyHttpUrl]


class ModelInstallJob(BaseModel):
    """Object that tracks the current status of an install request."""
    status: InstallStatus = Field(default=InstallStatus.WAITING, description="Current status of install process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Configuration metadata to apply to model before installing it")
    inplace: bool = Field(default=False, description="Leave model in its current location; otherwise install under models directory")
    source: ModelSource = Field(description="Source (URL, repo_id, or local path) of model")
    local_path: Path = Field(description="Path to locally-downloaded model; may be the same as the source")
    key: str = Field(default="<NO KEY>", description="After model is installed, this is its config record key")
    error_type: str = Field(default="", description="Class name of the exception that led to status==ERROR")
    error: str = Field(default="", description="Error traceback")  # noqa #501

    def set_error(self, e: Exception) -> None:
        """Record the error and traceback from an exception."""
        self.error_type = e.__class__.__name__
        self.error = traceback.format_exc()
        self.status = InstallStatus.ERROR


class ModelInstallServiceBase(ABC):
    """Abstract base class for InvokeAI model installation."""

    @abstractmethod
    def __init__(
            self,
            app_config: InvokeAIAppConfig,
            record_store: ModelRecordServiceBase,
            event_bus: Optional["EventServiceBase"] = None,
    ):
        """
        Create ModelInstallService object.

        :param config: Systemwide InvokeAIAppConfig.
        :param store: Systemwide ModelConfigStore
        :param event_bus: InvokeAI event bus for reporting events to.
        """

    @property
    @abstractmethod
    def app_config(self) -> InvokeAIAppConfig:
        """Return the appConfig object associated with the installer."""

    @property
    @abstractmethod
    def record_store(self) -> ModelRecordServiceBase:
        """Return the ModelRecoreService object associated with the installer."""

    @property
    @abstractmethod
    def event_bus(self) -> Optional[EventServiceBase]:
        """Return the event service base object associated with the installer."""

    @abstractmethod
    def register_path(
            self,
            model_path: Union[Path, str],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Probe and register the model at model_path.

        This keeps the model in its current location.

        :param model_path: Filesystem Path to the model.
        :param metadata: Dict of attributes that will override autoassigned values.
        :returns id: The string ID of the registered model.
        """

    @abstractmethod
    def unregister(self, key: str) -> None:
        """Remove model with indicated key from the database."""

    @abstractmethod
    def delete(self, key: str) -> None:  # noqa D102
        """Remove model with indicated key from the database. Delete its files only if they are within our models directory."""

    @abstractmethod
    def unconditionally_delete(self, key: str) -> None:
        """Remove model with indicated key from the database and unconditionally delete weight files from disk."""

    @abstractmethod
    def install_path(
            self,
            model_path: Union[Path, str],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Probe, register and install the model in the models directory.

        This moves the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :param metadata: Dict of attributes that will override autoassigned values.
        :returns id: The string ID of the registered model.
        """

    @abstractmethod
    def import_model(
            self,
            source: Union[str, Path, AnyHttpUrl],
            inplace: bool = True,
            variant: Optional[str] = None,
            subfolder: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            access_token: Optional[str] = None,
    ) -> ModelInstallJob:
        """Install the indicated model.

        :param source: Either a URL or a HuggingFace repo_id.

        :param inplace: If True, local paths will not be moved into
         the models directory, but registered in place (the default).

        :param variant: For HuggingFace models, this optional parameter
         specifies which variant to download (e.g. 'fp16')

        :param subfolder: When downloading HF repo_ids this can be used to
         specify a subfolder of the HF repository to download from.

        :param metadata: Optional dict. Any fields in this dict
         will override corresponding autoassigned probe fields. Use it to override
         `name`, `description`, `base_type`, `model_type`, `format`,
         `prediction_type`, `image_size`, and/or `ztsnr_training`.

        :param access_token: Access token for use in downloading remote
         models.

        This will download the model located at `source`,
        probe it, and install it into the models directory.
        This call is executed asynchronously in a separate
        thread and will issue the following events on the event bus:

             - model_install_started
             - model_install_error
             - model_install_completed

        The `inplace` flag does not affect the behavior of downloaded
        models, which are always moved into the `models` directory.

        The call returns a ModelInstallJob object which can be
        polled to learn the current status and/or error message.

        Variants recognized by HuggingFace currently are:
        1. onnx
        2. openvino
        3. fp16
        4. None (usually returns fp32 model)

        """

    @abstractmethod
    def get_job(self, source: ModelSource) -> ModelInstallJob:
        """Return the ModelInstallJob corresponding to the provided source."""

    @abstractmethod
    def list_jobs(self, source: Optional[ModelSource]=None) -> List[ModelInstallJob]:  # noqa D102
        """
        List active and complete install jobs.

        :param source: Filter by jobs whose sources are a partial match to the argument.
        """

    @abstractmethod
    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""

    @abstractmethod
    def wait_for_installs(self) -> Dict[Union[str, Path, AnyHttpUrl], ModelInstallJob]:
        """
        Wait for all pending installs to complete.

        This will block until all pending downloads have
        completed, been cancelled, or errored out. It will
        block indefinitely if one or more jobs are in the
        paused state.

        It will return a dict that maps the source model
        path, URL or repo_id to the ID of the installed model.
        """

    @abstractmethod
    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:
        """
        Recursively scan directory for new models and register or install them.

        :param scan_dir: Path to the directory to scan.
        :param install: Install if True, otherwise register in place.
        :returns list of IDs: Returns list of IDs of models registered/installed
        """

    @abstractmethod
    def sync_to_config(self) -> None:
        """Synchronize models on disk to those in the model record database."""
