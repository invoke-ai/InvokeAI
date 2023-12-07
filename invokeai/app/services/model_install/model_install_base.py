import re
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import Body
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from pydantic.networks import AnyHttpUrl
from typing_extensions import Annotated

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.backend.model_manager import AnyModelConfig


class InstallStatus(str, Enum):
    """State of an install job running in the background."""

    WAITING = "waiting"  # waiting to be dequeued
    RUNNING = "running"  # being processed
    COMPLETED = "completed"  # finished running
    ERROR = "error"  # terminated with an error message


class UnknownInstallJobException(Exception):
    """Raised when the status of an unknown job is requested."""


class StringLikeSource(BaseModel):
    """
    Base class for model sources, implements functions that lets the source be sorted and indexed.

    These shenanigans let this stuff work:

      source1 = LocalModelSource(path='C:/users/mort/foo.safetensors')
      mydict = {source1: 'model 1'}
      assert mydict['C:/users/mort/foo.safetensors'] == 'model 1'
      assert mydict[LocalModelSource(path='C:/users/mort/foo.safetensors')] == 'model 1'

      source2 = LocalModelSource(path=Path('C:/users/mort/foo.safetensors'))
      assert source1 == source2
      assert source1 == 'C:/users/mort/foo.safetensors'
    """

    def __hash__(self) -> int:
        """Return hash of the path field, for indexing."""
        return hash(str(self))

    def __lt__(self, other: object) -> int:
        """Return comparison of the stringified version, for sorting."""
        return str(self) < str(other)

    def __eq__(self, other: object) -> bool:
        """Return equality on the stringified version."""
        if isinstance(other, Path):
            return str(self) == other.as_posix()
        else:
            return str(self) == str(other)


class LocalModelSource(StringLikeSource):
    """A local file or directory path."""

    path: str | Path
    inplace: Optional[bool] = False
    type: Literal["local"] = "local"

    # these methods allow the source to be used in a string-like way,
    # for example as an index into a dict
    def __str__(self) -> str:
        """Return string version of path when string rep needed."""
        return Path(self.path).as_posix()


class HFModelSource(StringLikeSource):
    """A HuggingFace repo_id, with optional variant and sub-folder."""

    repo_id: str
    variant: Optional[str] = None
    subfolder: Optional[str | Path] = None
    access_token: Optional[str] = None
    type: Literal["hf"] = "hf"

    @field_validator("repo_id")
    @classmethod
    def proper_repo_id(cls, v: str) -> str:  # noqa D102
        if not re.match(r"^([.\w-]+/[.\w-]+)$", v):
            raise ValueError(f"{v}: invalid repo_id format")
        return v

    def __str__(self) -> str:
        """Return string version of repoid when string rep needed."""
        base: str = self.repo_id
        base += f":{self.subfolder}" if self.subfolder else ""
        base += f" ({self.variant})" if self.variant else ""
        return base


class URLModelSource(StringLikeSource):
    """A generic URL point to a checkpoint file."""

    url: AnyHttpUrl
    access_token: Optional[str] = None
    type: Literal["generic_url"] = "generic_url"

    def __str__(self) -> str:
        """Return string version of the url when string rep needed."""
        return str(self.url)


# Body() is being applied here rather than Field() because otherwise FastAPI will
# refuse to generate a schema. Relevant links:
#
# "Model Manager Refactor Phase 1 - SQL-based config storage
#        https://github.com/invoke-ai/InvokeAI/pull/5039#discussion_r1389752119 (comment)
# Param: xyz can only be a request body, using Body() when using discriminated unions
#        https://github.com/tiangolo/fastapi/discussions/9761
# Body parameter cannot be a pydantic union anymore sinve v0.95
#       https://github.com/tiangolo/fastapi/discussions/9287

ModelSource = Annotated[Union[LocalModelSource, HFModelSource, URLModelSource], Body(discriminator="type")]
ModelSourceValidator = TypeAdapter(ModelSource)


class ModelInstallJob(BaseModel):
    """Object that tracks the current status of an install request."""

    status: InstallStatus = Field(default=InstallStatus.WAITING, description="Current status of install process")
    config_in: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration information (e.g. 'description') to apply to model."
    )
    config_out: Optional[AnyModelConfig] = Field(
        default=None, description="After successful installation, this will hold the configuration object."
    )
    inplace: bool = Field(
        default=False, description="Leave model in its current location; otherwise install under models directory"
    )
    source: ModelSource = Field(description="Source (URL, repo_id, or local path) of model")
    local_path: Path = Field(description="Path to locally-downloaded model; may be the same as the source")
    error_type: Optional[str] = Field(default=None, description="Class name of the exception that led to status==ERROR")
    error: Optional[str] = Field(default=None, description="Error traceback")  # noqa #501

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

    def start(self, invoker: Invoker) -> None:
        """Call at InvokeAI startup time."""
        self.sync_to_config()

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
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Probe and register the model at model_path.

        This keeps the model in its current location.

        :param model_path: Filesystem Path to the model.
        :param config: Dict of attributes that will override autoassigned values.
        :returns id: The string ID of the registered model.
        """

    @abstractmethod
    def unregister(self, key: str) -> None:
        """Remove model with indicated key from the database."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove model with indicated key from the database. Delete its files only if they are within our models directory."""

    @abstractmethod
    def unconditionally_delete(self, key: str) -> None:
        """Remove model with indicated key from the database and unconditionally delete weight files from disk."""

    @abstractmethod
    def install_path(
        self,
        model_path: Union[Path, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Probe, register and install the model in the models directory.

        This moves the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :param config: Dict of attributes that will override autoassigned values.
        :returns id: The string ID of the registered model.
        """

    @abstractmethod
    def import_model(
        self,
        source: ModelSource,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelInstallJob:
        """Install the indicated model.

        :param source: ModelSource object

        :param config: Optional dict. Any fields in this dict
         will override corresponding autoassigned probe fields in the
         model's config record. Use it to override
         `name`, `description`, `base_type`, `model_type`, `format`,
         `prediction_type`, `image_size`, and/or `ztsnr_training`.

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
    def get_job(self, source: ModelSource) -> List[ModelInstallJob]:
        """Return the ModelInstallJob(s) corresponding to the provided source."""

    @abstractmethod
    def list_jobs(self) -> List[ModelInstallJob]:  # noqa D102
        """
        List active and complete install jobs.
        """

    @abstractmethod
    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""

    @abstractmethod
    def wait_for_installs(self) -> List[ModelInstallJob]:
        """
        Wait for all pending installs to complete.

        This will block until all pending installs have
        completed, been cancelled, or errored out. It will
        block indefinitely if one or more jobs are in the
        paused state.

        It will return the current list of jobs.
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
