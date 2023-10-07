# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import Field, parse_obj_as
from pydantic.networks import AnyHttpUrl

from invokeai.backend import get_precision
from invokeai.backend.model_manager import ModelConfigBase, ModelSearch
from invokeai.backend.model_manager.download import DownloadJobBase
from invokeai.backend.model_manager.install import ModelInstall, ModelInstallBase, ModelInstallJob
from invokeai.backend.model_manager.merge import MergeInterpolationMethod, ModelMerger
from invokeai.backend.util.logging import InvokeAILogger

from .config import InvokeAIAppConfig
from .events import EventServiceBase
from .model_record_service import ModelRecordServiceBase


class ModelInstallServiceBase(ModelInstallBase):  # This is an ABC
    """Responsible for downloading, installing and deleting models."""

    @abstractmethod
    def __init__(
        self, config: InvokeAIAppConfig, store: ModelRecordServiceBase, event_bus: Optional[EventServiceBase] = None
    ):
        """
        Initialize a ModelInstallService instance.

        :param config: InvokeAIAppConfig object
        :param store: A ModelRecordServiceBase object install to
        :param event_bus: Optional EventServiceBase object. If provided,
        installation and download events will be sent to the event bus as "model_event".
        """
        pass

    @abstractmethod
    def install_model(
        self,
        source: Union[str, Path, AnyHttpUrl],
        priority: int = 10,
        model_attributes: Optional[Dict[str, Any]] = None,
    ) -> ModelInstallJob:
        """Import a path, repo_id or URL. Returns an ModelInstallJob.

        :param model_attributes: Additional attributes to supplement/override
        the model information gained from automated probing.
        :param priority: Queue priority. Lower values have higher priority.

        Typical usage:
        job = model_manager.install(
                   'stabilityai/stable-diffusion-2-1',
                   model_attributes={'prediction_type": 'v-prediction'}
        )

        The result is an ModelInstallJob object, which provides
        information on the asynchronous model download and install
        process. A series of "install_model_event" events will be emitted
        until the install is completed, cancelled or errors out.
        """
        pass

    @abstractmethod
    def list_install_jobs(self) -> List[ModelInstallJob]:
        """Return a series of active or enqueued ModelInstallJobs."""
        pass

    @abstractmethod
    def id_to_job(self, id: int) -> ModelInstallJob:
        """Return the ModelInstallJob instance corresponding to the given job ID."""
        pass

    @abstractmethod
    def start_job(self, job_id: int):
        """Start the given install job if it is paused or idle."""
        pass

    @abstractmethod
    def pause_job(self, job_id: int):
        """Pause the given install job if it is paused or idle."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: int):
        """Cancel the given install job."""
        pass

    @abstractmethod
    def cancel_all_jobs(self):
        """Cancel all active jobs."""
        pass

    @abstractmethod
    def prune_jobs(self):
        """Remove completed or errored install jobs."""
        pass

    @abstractmethod
    def change_job_priority(self, job_id: int, delta: int):
        """
        Change an install job's priority.

        :param job_id: Job to change
        :param delta: Value to increment or decrement priority.

        Lower values are higher priority.  The default starting value is 10.
        Thus to make this a really high priority job:
           manager.change_job_priority(-10).
        """
        pass

    @abstractmethod
    def merge_models(
        self,
        model_keys: List[str] = Field(
            default=None, min_items=2, max_items=3, description="List of model keys to merge"
        ),
        merged_model_name: str = Field(default=None, description="Name of destination model after merging"),
        alpha: Optional[float] = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: Optional[bool] = False,
        merge_dest_directory: Optional[Path] = None,
    ) -> ModelConfigBase:
        """
        Merge two to three diffusrs pipeline models and save as a new model.

        :param model_keys: List of 2-3 model unique keys to merge
        :param merged_model_name: Name of destination merged model
        :param alpha: Alpha strength to apply to 2d and 3d model
        :param interp: Interpolation method. None (default)
        :param merge_dest_directory: Save the merged model to the designated directory (with 'merged_model_name' appended)
        """
        pass

    @abstractmethod
    def list_checkpoint_configs(self) -> List[Path]:
        """List the checkpoint config paths from ROOT/configs/stable-diffusion."""
        pass

    @abstractmethod
    def search_for_models(self, directory: Path) -> Set[Path]:
        """Return list of all models found in the designated directory."""
        pass


# implementation
class ModelInstallService(ModelInstall, ModelInstallServiceBase):
    """Responsible for managing models on disk and in memory."""

    _precision: Literal["float16", "float32"] = Field(description="Floating point precision, string form")
    _event_bus: Optional[EventServiceBase] = Field(description="an event bus to send install events to", default=None)

    def __init__(
        self, config: InvokeAIAppConfig, store: ModelRecordServiceBase, event_bus: Optional[EventServiceBase] = None
    ):
        """
        Initialize a ModelInstallService instance.

        :param config: InvokeAIAppConfig object
        :param store: Either a ModelRecordService object or a ModelConfigStore
        :param event_bus: Optional EventServiceBase object. If provided,

        Installation and download events will be sent to the event bus as "model_event".
        """
        self._event_bus = event_bus
        kwargs: Dict[str, Any] = {}
        if self._event_bus:
            kwargs.update(event_handlers=[self._event_bus.emit_model_event])
        self._precision = get_precision()
        logger = InvokeAILogger.get_logger()
        super().__init__(store=store, config=config, logger=logger, **kwargs)

    def start(self, invoker: Any):  # Because .processor is giving circular import errors, declaring invoker an 'Any'
        """Call automatically at process start."""
        self.scan_models_directory()  # synchronize new/deleted models found in models directory
        if autoimport := self._app_config.autoimport_dir:
            self._logger.info("Scanning autoimport directory for new models")
            self.scan_directory(self._app_config.root_path / autoimport)

    def install_model(
        self,
        source: Union[str, Path, AnyHttpUrl],
        priority: int = 10,
        model_attributes: Optional[Dict[str, Any]] = None,
    ) -> ModelInstallJob:
        """
        Add a model using a path, repo_id or URL.

        :param model_attributes: Dictionary of ModelConfigBase fields to
        attach to the model. When installing a URL or repo_id, some metadata
        values, such as `tags` will be automagically added.
        :param priority: Queue priority for this install job. Lower value jobs
        will run before higher value ones.
        """
        self.logger.debug(f"add model {source}")
        variant = "fp16" if self._precision == "float16" else None
        job = self.install(
            source,
            probe_override=model_attributes,
            variant=variant,
            priority=priority,
        )
        assert isinstance(job, ModelInstallJob)
        return job

    def list_install_jobs(self) -> List[ModelInstallJob]:
        """Return a series of active or enqueued ModelInstallJobs."""
        queue = self.queue
        jobs: List[DownloadJobBase] = queue.list_jobs()
        return [parse_obj_as(ModelInstallJob, x) for x in jobs]  # downcast to proper type

    def id_to_job(self, id: int) -> ModelInstallJob:
        """Return the ModelInstallJob instance corresponding to the given job ID."""
        job = self.queue.id_to_job(id)
        assert isinstance(job, ModelInstallJob)
        return job

    def start_job(self, job_id: int):
        """Start the given install job if it is paused or idle."""
        queue = self.queue
        queue.start_job(queue.id_to_job(job_id))

    def pause_job(self, job_id: int):
        """Pause the given install job if it is paused or idle."""
        queue = self.queue
        queue.pause_job(queue.id_to_job(job_id))

    def cancel_job(self, job_id: int):
        """Cancel the given install job."""
        queue = self.queue
        queue.cancel_job(queue.id_to_job(job_id))

    def cancel_all_jobs(self):
        """Cancel all active install job."""
        queue = self.queue
        queue.cancel_all_jobs()

    def prune_jobs(self):
        """Cancel all active install job."""
        queue = self.queue
        queue.prune_jobs()

    def change_job_priority(self, job_id: int, delta: int):
        """
        Change an install job's priority.

        :param job_id: Job to change
        :param delta: Value to increment or decrement priority.

        Lower values are higher priority.  The default starting value is 10.
        Thus to make this a really high priority job:
           manager.change_job_priority(-10).
        """
        queue = self.queue
        queue.change_priority(queue.id_to_job(job_id), delta)

    def del_model(
        self,
        key: str,
        delete_files: bool = False,
    ):
        """
        Delete the named model from configuration.

        If delete_files is true,
        then the underlying weight file or diffusers directory will be deleted
        as well.
        """
        model_info = self.store.get_model(key)
        self.logger.debug(f"delete model {model_info.name}")
        self.store.del_model(key)
        if delete_files and Path(self._app_config.models_path / model_info.path).exists():
            path = Path(model_info.path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def convert_model(
        self,
        key: str,
        dest_directory: Optional[Path] = None,
    ) -> ModelConfigBase:
        """
        Convert a checkpoint file into a diffusers folder.

        Delete the cached
        version and delete the original checkpoint file if it is in the models
        directory.

        :param key: Key of the model to convert
        :param convert_dest_directory: Save the converted model to the designated directory (`models/etc/etc` by default)

        This will raise a ValueError unless the model is a checkpoint. It will
        also raise a ValueError in the event that there is a similarly-named diffusers
        directory already in place.
        """
        model_info = self.store.get_model(key)
        self.logger.info(f"Converting model {model_info.name} into a diffusers")
        return super().convert_model(key, dest_directory)

    @property
    def logger(self):
        """Get the logger associated with this instance."""
        return self._logger

    @property
    def store(self):
        """Get the store associated with this instance."""
        return self._store

    def merge_models(
        self,
        model_keys: List[str] = Field(
            default=None, min_items=2, max_items=3, description="List of model keys to merge"
        ),
        merged_model_name: Optional[str] = Field(default=None, description="Name of destination model after merging"),
        alpha: Optional[float] = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: Optional[bool] = False,
        merge_dest_directory: Optional[Path] = None,
    ) -> ModelConfigBase:
        """
        Merge two to three diffusrs pipeline models and save as a new model.

        :param model_keys: List of 2-3 model unique keys to merge
        :param merged_model_name: Name of destination merged model
        :param alpha: Alpha strength to apply to 2d and 3d model
        :param interp: Interpolation method. None (default)
        :param merge_dest_directory: Save the merged model to the designated directory (with 'merged_model_name' appended)
        """
        merger = ModelMerger(self.store)
        try:
            if not merged_model_name:
                merged_model_name = "+".join([self.store.get_model(x).name for x in model_keys])
                raise Exception("not implemented")

            result = merger.merge_diffusion_models_and_save(
                model_keys=model_keys,
                merged_model_name=merged_model_name,
                alpha=alpha,
                interp=interp,
                force=force,
                merge_dest_directory=merge_dest_directory,
            )
        except AssertionError as e:
            raise ValueError(e)
        return result

    def search_for_models(self, directory: Path) -> Set[Path]:
        """
        Return list of all models found in the designated directory.

        :param directory: Path to the directory to recursively search.
        returns a list of model paths
        """
        return ModelSearch().search(directory)

    def list_checkpoint_configs(self) -> List[Path]:
        """List the checkpoint config paths from ROOT/configs/stable-diffusion."""
        config = self._app_config
        conf_path = config.legacy_conf_path
        root_path = config.root_path
        return [(conf_path / x).relative_to(root_path) for x in conf_path.glob("**/*.yaml")]
