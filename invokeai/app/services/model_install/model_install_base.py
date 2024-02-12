# Copyright 2023 Lincoln D. Stein and the InvokeAI development team
"""Baseclass definitions for the model installer."""

import re
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic.networks import AnyHttpUrl
from typing_extensions import Annotated

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadJob, DownloadQueueServiceBase
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.backend.model_manager import AnyModelConfig, ModelRepoVariant
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata, ModelMetadataStore


class InstallStatus(str, Enum):
    """State of an install job running in the background."""

    WAITING = "waiting"  # waiting to be dequeued
    DOWNLOADING = "downloading"  # downloading of model files in process
    RUNNING = "running"  # being processed
    COMPLETED = "completed"  # finished running
    ERROR = "error"  # terminated with an error message
    CANCELLED = "cancelled"  # terminated with an error message


class ModelInstallPart(BaseModel):
    url: AnyHttpUrl
    path: Path
    bytes: int = 0
    total_bytes: int = 0


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


class CivitaiModelSource(StringLikeSource):
    """A Civitai version id, with optional variant and access token."""

    version_id: int
    variant: Optional[ModelRepoVariant] = None
    access_token: Optional[str] = None
    type: Literal["civitai"] = "civitai"

    def __str__(self) -> str:
        """Return string version of repoid when string rep needed."""
        base: str = str(self.version_id)
        base += f" ({self.variant})" if self.variant else ""
        return base


class HFModelSource(StringLikeSource):
    """
    A HuggingFace repo_id with optional variant, sub-folder and access token.
    Note that the variant option, if not provided to the constructor, will default to fp16, which is
    what people (almost) always want.
    """

    repo_id: str
    variant: Optional[ModelRepoVariant] = ModelRepoVariant.FP16
    subfolder: Optional[Path] = None
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
        base += f":{self.variant or ''}"
        base += f":{self.subfolder}" if self.subfolder else ""
        return base


class URLModelSource(StringLikeSource):
    """A generic URL point to a checkpoint file."""

    url: AnyHttpUrl
    access_token: Optional[str] = None
    type: Literal["url"] = "url"

    def __str__(self) -> str:
        """Return string version of the url when string rep needed."""
        return str(self.url)


ModelSource = Annotated[
    Union[LocalModelSource, HFModelSource, CivitaiModelSource, URLModelSource], Field(discriminator="type")
]


class ModelInstallJob(BaseModel):
    """Object that tracks the current status of an install request."""

    id: int = Field(description="Unique ID for this job")
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
    bytes: int = Field(
        default=0, description="For a remote model, the number of bytes downloaded so far (may not be available)"
    )
    total_bytes: int = Field(default=0, description="Total size of the model to be installed")
    source_metadata: Optional[AnyModelRepoMetadata] = Field(
        default=None, description="Metadata provided by the model source"
    )
    download_parts: Set[DownloadJob] = Field(
        default_factory=set, description="Download jobs contributing to this install"
    )
    # internal flags and transitory settings
    _install_tmpdir: Optional[Path] = PrivateAttr(default=None)
    _exception: Optional[Exception] = PrivateAttr(default=None)

    def set_error(self, e: Exception) -> None:
        """Record the error and traceback from an exception."""
        self._exception = e
        self.status = InstallStatus.ERROR

    def cancel(self) -> None:
        """Call to cancel the job."""
        self.status = InstallStatus.CANCELLED

    @property
    def error_type(self) -> Optional[str]:
        """Class name of the exception that led to status==ERROR."""
        return self._exception.__class__.__name__ if self._exception else None

    @property
    def error(self) -> Optional[str]:
        """Error traceback."""
        return "".join(traceback.format_exception(self._exception)) if self._exception else None

    @property
    def cancelled(self) -> bool:
        """Set status to CANCELLED."""
        return self.status == InstallStatus.CANCELLED

    @property
    def errored(self) -> bool:
        """Return true if job has errored."""
        return self.status == InstallStatus.ERROR

    @property
    def waiting(self) -> bool:
        """Return true if job is waiting to run."""
        return self.status == InstallStatus.WAITING

    @property
    def downloading(self) -> bool:
        """Return true if job is downloading."""
        return self.status == InstallStatus.DOWNLOADING

    @property
    def running(self) -> bool:
        """Return true if job is running."""
        return self.status == InstallStatus.RUNNING

    @property
    def complete(self) -> bool:
        """Return true if job completed without errors."""
        return self.status == InstallStatus.COMPLETED

    @property
    def in_terminal_state(self) -> bool:
        """Return true if job is in a terminal state."""
        return self.status in [InstallStatus.COMPLETED, InstallStatus.ERROR, InstallStatus.CANCELLED]


class ModelInstallServiceBase(ABC):
    """Abstract base class for InvokeAI model installation."""

    @abstractmethod
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        metadata_store: ModelMetadataStore,
        event_bus: Optional["EventServiceBase"] = None,
    ):
        """
        Create ModelInstallService object.

        :param config: Systemwide InvokeAIAppConfig.
        :param store: Systemwide ModelConfigStore
        :param event_bus: InvokeAI event bus for reporting events to.
        """

    # make the invoker optional here because we don't need it and it
    # makes the installer harder to use outside the web app
    @abstractmethod
    def start(self, invoker: Optional[Invoker] = None) -> None:
        """Start the installer service."""

    @abstractmethod
    def stop(self, invoker: Optional[Invoker] = None) -> None:
        """Stop the model install service. After this the objection can be safely deleted."""

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
    def heuristic_import(
        self,
        source: str,
        config: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
    ) -> ModelInstallJob:
        r"""Install the indicated model using heuristics to interpret user intentions.

        :param source: String source
        :param config: Optional dict. Any fields in this dict
         will override corresponding autoassigned probe fields in the
         model's config record as described in `import_model()`.
        :param access_token: Optional access token for remote sources.

        The source can be:
        1. A local file path in posix() format (`/foo/bar` or `C:\foo\bar`)
        2. An http or https URL (`https://foo.bar/foo`)
        3. A HuggingFace repo_id (`foo/bar`, `foo/bar:fp16`, `foo/bar:fp16:vae`)

        We extend the HuggingFace repo_id syntax to include the variant and the
        subfolder or path. The following are acceptable alternatives:
            stabilityai/stable-diffusion-v4
            stabilityai/stable-diffusion-v4:fp16
            stabilityai/stable-diffusion-v4:fp16:vae
            stabilityai/stable-diffusion-v4::/checkpoints/sd4.safetensors
            stabilityai/stable-diffusion-v4:onnx:vae

        Because a local file path can look like a huggingface repo_id, the logic
        first checks whether the path exists on disk, and if not, it is treated as
        a parseable huggingface repo.

        The previous support for recursing into a local folder and loading all model-like files
        has been removed.
        """
        pass

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
    def get_job_by_source(self, source: ModelSource) -> List[ModelInstallJob]:
        """Return the ModelInstallJob(s) corresponding to the provided source."""

    @abstractmethod
    def get_job_by_id(self, id: int) -> ModelInstallJob:
        """Return the ModelInstallJob corresponding to the provided id. Raises ValueError if no job has that ID."""

    @abstractmethod
    def list_jobs(self) -> List[ModelInstallJob]:  # noqa D102
        """
        List active and complete install jobs.
        """

    @abstractmethod
    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""

    @abstractmethod
    def cancel_job(self, job: ModelInstallJob) -> None:
        """Cancel the indicated job."""

    @abstractmethod
    def wait_for_installs(self, timeout: int = 0) -> List[ModelInstallJob]:
        """
        Wait for all pending installs to complete.

        This will block until all pending installs have
        completed, been cancelled, or errored out.

        :param timeout: Wait up to indicated number of seconds. Raise an Exception('timeout') if
        installs do not complete within the indicated time.
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
