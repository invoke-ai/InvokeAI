# Copyright 2023 Lincoln D. Stein and the InvokeAI development team
"""Baseclass definitions for the model installer."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.model_install_common import ModelInstallJob, ModelSource
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceBase
from invokeai.backend.model_manager import AnyModelConfig


class ModelInstallServiceBase(ABC):
    """Abstract base class for InvokeAI model installation."""

    @abstractmethod
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
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
    def event_bus(self) -> Optional["EventServiceBase"]:
        """Return the event service base object associated with the installer."""

    @abstractmethod
    def register_path(
        self,
        model_path: Union[Path, str],
        config: Optional[ModelRecordChanges] = None,
    ) -> str:
        """
        Probe and register the model at model_path.

        This keeps the model in its current location.

        :param model_path: Filesystem Path to the model.
        :param config: ModelRecordChanges object that will override autoassigned model record values.
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
        config: Optional[ModelRecordChanges] = None,
    ) -> str:
        """
        Probe, register and install the model in the models directory.

        This moves the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :param config: ModelRecordChanges object that will override autoassigned model record values.
        :returns id: The string ID of the registered model.
        """

    @abstractmethod
    def heuristic_import(
        self,
        source: str,
        config: Optional[ModelRecordChanges] = None,
        access_token: Optional[str] = None,
        inplace: Optional[bool] = False,
    ) -> ModelInstallJob:
        r"""Install the indicated model using heuristics to interpret user intentions.

        :param source: String source
        :param config: Optional ModelRecordChanges object. Any fields in this object
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
        config: Optional[ModelRecordChanges] = None,
    ) -> ModelInstallJob:
        """Install the indicated model.

        :param source: ModelSource object

        :param config: Optional dict. Any fields in this dict
         will override corresponding autoassigned probe fields in the
         model's config record. Use it to override
         `name`, `description`, `base_type`, `model_type`, `format`,
         `prediction_type`, and/or `image_size`.

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
    def wait_for_job(self, job: ModelInstallJob, timeout: int = 0) -> ModelInstallJob:
        """Wait for the indicated job to reach a terminal state.

        This will block until the indicated install job has completed,
        been cancelled, or errored out.

        :param job: The job to wait on.
        :param timeout: Wait up to indicated number of seconds. Raise a TimeoutError if
        the job hasn't completed within the indicated time.
        """

    @abstractmethod
    def wait_for_installs(self, timeout: int = 0) -> List[ModelInstallJob]:
        """
        Wait for all pending installs to complete.

        This will block until all pending installs have
        completed, been cancelled, or errored out.

        :param timeout: Wait up to indicated number of seconds. Raise an Exception('timeout') if
        installs do not complete within the indicated time. A timeout of zero (the default)
        will block indefinitely until the installs complete.
        """

    @abstractmethod
    def sync_model_path(self, key: str) -> AnyModelConfig:
        """
        Move model into the location indicated by its basetype, type and name.

        Call this after updating a model's attributes in order to move
        the model's path into the location indicated by its basetype, type and
        name. Applies only to models whose paths are within the root `models_dir`
        directory.

        May raise an UnknownModelException.
        """

    @abstractmethod
    def download_and_cache_model(self, source: str | AnyHttpUrl) -> Path:
        """
        Download the model file located at source to the models cache and return its Path.

        :param source: A string representing a URL or repo_id.

        The model file will be downloaded into the system-wide model cache
        (`models/.cache`) if it isn't already there. Note that the model cache
        is periodically cleared of infrequently-used entries when the model
        converter runs.

        Note that this doesn't automatically install or register the model, but is
        intended for use by nodes that need access to models that aren't directly
        supported by InvokeAI. The downloading process takes advantage of the download queue
        to avoid interrupting other operations.
        """
