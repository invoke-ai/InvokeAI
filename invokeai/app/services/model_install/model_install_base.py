from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union
from pydantic import BaseModel, Field
from pydantic.networks import AnyHttpUrl

from invokeai.app.services.model_records import  ModelRecordServiceBase
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events import EventServiceBase


class InstallStatus(str, Enum):
    """State of an install job running in the background."""

    WAITING = "waiting"      # waiting to be dequeued
    COMPLETED = "completed"  # finished running
    ERROR = "error"          # terminated with an error message


class ModelInstallStatus(BaseModel):
    status: InstallStatus = Field(default=InstallStatus.WAITING, description="Current status of install process")  # noqa #501
    error: Optional[Exception] = Field(default=None, description="Exception that led to status==ERROR")  # noqa #501


class ModelInstallServiceBase(ABC):
    """Abstract base class for InvokeAI model installation."""

    @abstractmethod
    def __init__(
            self,
            config: InvokeAIAppConfig,
            store: ModelRecordServiceBase,
            event_bus: Optional["EventServiceBase"] = None,
    ):
        """
        Create ModelInstallService object.

        :param config: Systemwide InvokeAIAppConfig.
        :param store: Systemwide ModelConfigStore
        :param event_bus: InvokeAI event bus for reporting events to.
        """
        pass

    @property
    @abstractmethod
    def config(self) -> InvokeAIAppConfig:
        """Return the app_config used by the installer."""
        pass

    @property
    @abstractmethod
    def store(self) -> ModelRecordServiceBase:
        """Return the storage backend used by the installer."""
        pass

    @abstractmethod
    def register_path(
            self,
            model_path: Union[Path, str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Probe and register the model at model_path.

        This keeps the model in its current location.

        :param model_path: Filesystem Path to the model.
        :param name: Name for the model (optional)
        :param description: Description for the model (optional)
        :param metadata: Dict of attributes that will override probed values.
        :returns id: The string ID of the registered model.
        """
        pass

    @abstractmethod
    def install_path(
            self,
            model_path: Union[Path, str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    )-> str:
        """
        Probe, register and install the model in the models directory.

        This moves the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :param name: Name for the model (optional)
        :param description: Description for the model (optional)
        :param metadata: Dict of attributes that will override probed values.
        :returns id: The string ID of the registered model.
        """
        pass

    @abstractmethod
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
    ) -> ModelInstallStatus:
        """Install the indicated model.

        :param source: Either a URL or a HuggingFace repo_id.

        :param inplace: If True, local paths will not be moved into
         the models directory, but registered in place (the default).

        :param variant: For HuggingFace models, this optional parameter
         specifies which variant to download (e.g. 'fp16')

        :param subfolder: When downloading HF repo_ids this can be used to
         specify a subfolder of the HF repository to download from.

        :param metadata: Optional dict. Any fields in this dict
         will override corresponding probe fields. Use it to override
         `base_type`, `model_type`, `format`, `prediction_type`, `image_size`,
         and `ztsnr_training`.

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

        The call returns a ModelInstallStatus object which can be
        polled to learn the current status and/or error message.

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
    def sync_to_config(self):
        """Synchronize models on disk to those in memory."""
        pass

    @abstractmethod
    def hash(self, model_path: Union[Path, str]) -> str:
        """
        Compute and return the fast hash of the model.

        :param model_path: Path to the model on disk.
        :return str: FastHash of the model for use as an ID.
        """
        pass

    # The following are internal methods
    @abstractmethod
    def _create_name(self, model_path: Union[Path, str]) -> str:
        """
        Creates a default name for the model.

        :param model_path: Path to the model on disk.
        :return str: Model name
        """
        pass

    @abstractmethod
    def _create_description(self, model_path: Union[Path, str]) -> str:
        """
        Creates a default description for the model.

        :param model_path: Path to the model on disk.
        :return str: Model description
        """
        pass

    @abstractmethod
    def _create_id(self, model_path: Union[Path, str], name: Optional[str] = None) -> str:
        """
        Creates a unique ID for the model for use with the model records module.  # noqa E501

        :param model_path: Path to the model on disk.
        :param name: (optional) non-default name for the model
        :return str: Unique ID
        """
        pass
