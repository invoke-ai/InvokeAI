import re
import traceback
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Set, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic.networks import AnyHttpUrl
from typing_extensions import Annotated

from invokeai.app.services.download import DownloadJob, MultiFileDownloadJob
from invokeai.app.services.model_records import ModelRecordChanges
from invokeai.backend.model_manager.config import AnyModelConfig
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant, ModelSourceType


class InstallStatus(str, Enum):
    """State of an install job running in the background."""

    WAITING = "waiting"  # waiting to be dequeued
    DOWNLOADING = "downloading"  # downloading of model files in process
    DOWNLOADS_DONE = "downloads_done"  # downloading done, waiting to run
    RUNNING = "running"  # being processed
    COMPLETED = "completed"  # finished running
    ERROR = "error"  # terminated with an error message
    CANCELLED = "cancelled"  # terminated with an error message


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
        if self.variant:
            base += f":{self.variant or ''}"
        if self.subfolder:
            base += f"::{self.subfolder.as_posix()}"
        return base


class URLModelSource(StringLikeSource):
    """A generic URL point to a checkpoint file."""

    url: AnyHttpUrl
    access_token: Optional[str] = None
    type: Literal["url"] = "url"

    def __str__(self) -> str:
        """Return string version of the url when string rep needed."""
        return str(self.url)


ModelSource = Annotated[Union[LocalModelSource, HFModelSource, URLModelSource], Field(discriminator="type")]

MODEL_SOURCE_TO_TYPE_MAP = {
    URLModelSource: ModelSourceType.Url,
    HFModelSource: ModelSourceType.HFRepoID,
    LocalModelSource: ModelSourceType.Path,
}


class ModelInstallJob(BaseModel):
    """Object that tracks the current status of an install request."""

    id: int = Field(description="Unique ID for this job")
    status: InstallStatus = Field(default=InstallStatus.WAITING, description="Current status of install process")
    error_reason: Optional[str] = Field(default=None, description="Information about why the job failed")
    config_in: ModelRecordChanges = Field(
        default_factory=ModelRecordChanges,
        description="Configuration information (e.g. 'description') to apply to model.",
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
    error: Optional[str] = Field(
        default=None, description="On an error condition, this field will contain the text of the exception"
    )
    error_traceback: Optional[str] = Field(
        default=None, description="On an error condition, this field will contain the exception traceback"
    )
    # internal flags and transitory settings
    _install_tmpdir: Optional[Path] = PrivateAttr(default=None)
    _multifile_job: Optional[MultiFileDownloadJob] = PrivateAttr(default=None)
    _exception: Optional[Exception] = PrivateAttr(default=None)

    def set_error(self, e: Exception) -> None:
        """Record the error and traceback from an exception."""
        self._exception = e
        self.error = str(e)
        self.error_traceback = self._format_error(e)
        self.status = InstallStatus.ERROR
        self.error_reason = self._exception.__class__.__name__ if self._exception else None

    def cancel(self) -> None:
        """Call to cancel the job."""
        self.status = InstallStatus.CANCELLED

    @property
    def error_type(self) -> Optional[str]:
        """Class name of the exception that led to status==ERROR."""
        return self._exception.__class__.__name__ if self._exception else None

    def _format_error(self, exception: Exception) -> str:
        """Error traceback."""
        return "".join(traceback.format_exception(exception))

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
    def downloads_done(self) -> bool:
        """Return true if job's downloads ae done."""
        return self.status == InstallStatus.DOWNLOADS_DONE

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
