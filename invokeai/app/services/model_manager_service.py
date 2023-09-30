# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, parse_obj_as
from pydantic.networks import AnyHttpUrl

from invokeai.app.models.exceptions import CanceledException
from invokeai.backend.model_manager import (
    BaseModelType,
    DuplicateModelException,
    ModelConfigBase,
    ModelInfo,
    ModelInstallJob,
    ModelLoad,
    ModelSearch,
    ModelType,
    SubModelType,
    UnknownModelException,
)
from invokeai.backend.model_manager.cache import CacheStats
from invokeai.backend.model_manager.download import DownloadJobBase
from invokeai.backend.model_manager.merge import MergeInterpolationMethod, ModelMerger

from .config import InvokeAIAppConfig
from .events import EventServiceBase

if TYPE_CHECKING:
    from ..invocations.baseinvocation import InvocationContext


class ModelManagerServiceBase(ABC):
    """Responsible for managing models on disk and in memory."""

    @abstractmethod
    def __init__(self, config: InvokeAIAppConfig, event_bus: Optional[EventServiceBase] = None):
        """
        Initialize a ModelManagerService.

        :param config: InvokeAIAppConfig object
        :param event_bus: Optional EventServiceBase object. If provided,
        installation and download events will be sent to the event bus.
        """
        pass

    @abstractmethod
    def get_model(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """Retrieve the indicated model identified by key.

        :param key: Unique key returned by the ModelConfigStore module.
        :param submodel_type: Submodel to return (required for main models)
        :param context" Optional InvocationContext, used in event reporting.
        """
        pass

    @property
    @abstractmethod
    def logger(self):
        pass

    @abstractmethod
    def model_exists(
        self,
        key: str,
    ) -> bool:
        pass

    @abstractmethod
    def model_info(self, key: str) -> ModelConfigBase:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        Uses the exact format as the omegaconf stanza.
        """
        pass

    @abstractmethod
    def list_models(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[ModelConfigBase]:
        """
        Return a list of ModelConfigBases that match the base, type and name criteria.
        :param base_model: Filter by the base model type.
        :param model_type: Filter by the model type.
        :param model_name: Filter by the model name.
        """
        pass

    def list_model(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> ModelConfigBase:
        """
        Return information about the model using the same format as list_models().

        If there are more than one model that match, raises a DuplicateModelException.
        If no model matches, raises an UnknownModelException
        """
        model_configs = self.list_models(model_name=model_name, base_model=base_model, model_type=model_type)
        if len(model_configs) > 1:
            raise DuplicateModelException(
                "More than one model share the same name and type: {base_model}/{model_type}/{model_name}"
            )
        if len(model_configs) == 0:
            raise UnknownModelException("No known model with name and type: {base_model}/{model_type}/{model_name}")
        return model_configs[0]

    def all_models(self) -> List[ModelConfigBase]:
        """Return a list of all the models."""
        return self.list_models()

    @abstractmethod
    def add_model(
        self, model_path: Path, probe_overrides: Optional[Dict[str, Any]] = None, wait: bool = False
    ) -> ModelInstallJob:
        """
        Add a model using its path, with a dictionary of attributes.

        Will fail with an assertion error if the name already exists.
        """
        pass

    @abstractmethod
    def update_model(
        self,
        key: str,
        new_config: Union[dict, ModelConfigBase],
    ) -> ModelConfigBase:
        """
        Update the named model with a dictionary of attributes. Will fail with a
        UnknownModelException if the name does not already exist.

        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model key is unknown.
        """
        pass

    @abstractmethod
    def del_model(self, key: str, delete_files: bool = False):
        """
        Delete the named model from configuration. If delete_files
        is true, then the underlying file or directory will be
        deleted as well.
        """
        pass

    def rename_model(
        self,
        key: str,
        new_name: str,
    ) -> ModelConfigBase:
        """
        Rename the indicated model.
        """
        return self.update_model(key, {"name": new_name})

    @abstractmethod
    def list_checkpoint_configs(self) -> List[Path]:
        """List the checkpoint config paths from ROOT/configs/stable-diffusion."""
        pass

    @abstractmethod
    def convert_model(
        self,
        key: str,
        convert_dest_directory: Path,
    ) -> ModelConfigBase:
        """
        Convert a checkpoint file into a diffusers folder.

        This will delete the cached version if there is any and delete the original
        checkpoint file if it is in the models directory.
        :param key: Unique key for the model to convert.
        :param convert_dest_directory: Save the converted model to the designated directory (`models/etc/etc` by default)

        This will raise a ValueError unless the model is not a checkpoint. It will
        also raise a ValueError in the event that there is a similarly-named diffusers
        directory already in place.
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
    def search_for_models(self, directory: Path) -> List[Path]:
        """
        Return list of all models found in the designated directory.
        """
        pass

    @abstractmethod
    def sync_to_config(self):
        """
        Synchronize the in-memory models with on-disk.

        Re-read models.yaml, rescan the models directory, and reimport models
        in the autoimport directories. Call after making changes outside the
        model manager API.
        """
        pass

    @abstractmethod
    def collect_cache_stats(self, cache_stats: CacheStats):
        """Reset model cache statistics for graph with graph_id."""
        pass


# implementation
class ModelManagerService(ModelManagerServiceBase):
    """Responsible for managing models on disk and in memory."""

    _loader: ModelLoad = Field(description="InvokeAIAppConfig object for the current process")
    _event_bus: Optional[EventServiceBase] = Field(description="an event bus to send install events to", default=None)

    def __init__(self, config: InvokeAIAppConfig, event_bus: Optional["EventServiceBase"] = None):
        """
        Initialize a ModelManagerService.

        :param config: InvokeAIAppConfig object
        :param event_bus: Optional EventServiceBase object. If provided,
        installation and download events will be sent to the event bus.
        """
        self._event_bus = event_bus
        kwargs: Dict[str, Any] = {}
        if self._event_bus:
            kwargs.update(event_handlers=[self._event_bus.emit_model_event])
        self._loader = ModelLoad(config, **kwargs)
        self._loader.installer.scan_models_directory()  # synchronize new/deleted models found in models directory

    def get_model(
        self,
        key: str,
        submodel_type: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """
        Retrieve the indicated model.

        The submodel is required when fetching a main model.
        """
        model_info: ModelInfo = self._loader.get_model(key, submodel_type)

        # we can emit model loading events if we are executing with access to the invocation context
        if context:
            self._emit_load_event(
                context=context,
                model_key=key,
                submodel=submodel_type,
                model_info=model_info,
            )

        return model_info

    def model_exists(
        self,
        key: str,
    ) -> bool:
        """
        Verify that a model with the given key exists.

        Given a model key, returns True if it is a valid
        identifier.
        """
        return self._loader.store.exists(key)

    def model_info(self, key: str) -> ModelConfigBase:
        """
        Return configuration information about a model.

        Given a model key returns the ModelConfigBase describing it.
        """
        return self._loader.store.get_model(key)

    # def all_models(self) -> List[ModelConfigBase]  -- defined in base class, same as list_models()
    # def list_model(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -- defined in base class

    def list_models(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[ModelConfigBase]:
        """
        Return a ModelConfigBase object for each model in the database.
        """
        return self._loader.store.search_by_name(model_name=model_name, base_model=base_model, model_type=model_type)

    def add_model(
        self, model_path: Path, model_attributes: Optional[dict] = None, wait: bool = False
    ) -> ModelInstallJob:
        """
        Add a model using its path, with a dictionary of attributes.

        Will fail with an
        assertion error if the name already exists.
        """
        self.logger.debug(f"add/update model {model_path}")
        return ModelInstallJob.parse_obj(
            self._loader.installer.install(
                model_path,
                probe_override=model_attributes,
            )
        )

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
        variant = "fp16" if self._loader.precision == "float16" else None
        return ModelInstallJob.parse_obj(
            self._loader.installer.install(
                source,
                probe_override=model_attributes,
                variant=variant,
                priority=priority,
            )
        )

    def list_install_jobs(self) -> List[ModelInstallJob]:
        """Return a series of active or enqueued ModelInstallJobs."""
        queue = self._loader.queue
        jobs: List[DownloadJobBase] = queue.list_jobs()
        return [parse_obj_as(ModelInstallJob, x) for x in jobs]  # downcast to proper type

    def id_to_job(self, id: int) -> ModelInstallJob:
        """Return the ModelInstallJob instance corresponding to the given job ID."""
        return ModelInstallJob.parse_obj(self._loader.queue.id_to_job(id))

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
        return self._loader.installer.wait_for_installs()

    def start_job(self, job_id: int):
        """Start the given install job if it is paused or idle."""
        queue = self._loader.queue
        queue.start_job(queue.id_to_job(job_id))

    def pause_job(self, job_id: int):
        """Pause the given install job if it is paused or idle."""
        queue = self._loader.queue
        queue.pause_job(queue.id_to_job(job_id))

    def cancel_job(self, job_id: int):
        """Cancel the given install job."""
        queue = self._loader.queue
        queue.cancel_job(queue.id_to_job(job_id))

    def cancel_all_jobs(self):
        """Cancel all active install job."""
        queue = self._loader.queue
        queue.cancel_all_jobs()

    def prune_jobs(self):
        """Cancel all active install job."""
        queue = self._loader.queue
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
        queue = self._loader.queue
        queue.change_priority(queue.id_to_job(job_id), delta)

    def update_model(
        self,
        key: str,
        new_config: Union[dict, ModelConfigBase],
    ) -> ModelConfigBase:
        """
        Update the named model with a dictionary of attributes.

        Will fail with a
        UnknownModelException if the name does not already exist.

        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model key is unknown.
        """
        self.logger.debug(f"update model {key}")
        new_info = self._loader.store.update_model(key, new_config)
        self._loader.installer.sync_model_path(new_info.key)
        return new_info

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
        model_info = self.model_info(key)
        self.logger.debug(f"delete model {model_info.name}")
        self._loader.store.del_model(key)
        if delete_files and Path(model_info.path).exists():
            path = Path(model_info.path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def convert_model(
        self,
        key: str,
        convert_dest_directory: Path,
    ) -> ModelConfigBase:
        """
        Convert a checkpoint file into a diffusers folder, deleting the cached
        version and deleting the original checkpoint file if it is in the models
        directory.

        :param key: Key of the model to convert
        :param convert_dest_directory: Save the converted model to the designated directory (`models/etc/etc` by default)

        This will raise a ValueError unless the model is not a checkpoint. It will
        also raise a ValueError in the event that there is a similarly-named diffusers
        directory already in place.
        """
        model_info = self.model_info(key)
        self.logger.info(f"Converting model {model_info.name} into a diffusers")
        return self._loader.installer.convert_model(key, convert_dest_directory)

    def collect_cache_stats(self, cache_stats: CacheStats):
        """
        Reset model cache statistics. Is this used?
        """
        self._loader.collect_cache_stats(cache_stats)

    def _emit_load_event(
        self,
        context: InvocationContext,
        model_key: str,
        submodel: Optional[SubModelType] = None,
        model_info: Optional[ModelInfo] = None,
    ):
        if context.services.queue.is_canceled(context.graph_execution_state_id):
            raise CanceledException()

        if model_info:
            context.services.events.emit_model_load_completed(
                queue_id=context.queue_id,
                queue_item_id=context.queue_item_id,
                queue_batch_id=context.queue_batch_id,
                graph_execution_state_id=context.graph_execution_state_id,
                model_key=model_key,
                submodel=submodel,
                model_info=model_info,
            )
        else:
            context.services.events.emit_model_load_started(
                queue_id=context.queue_id,
                queue_item_id=context.queue_item_id,
                queue_batch_id=context.queue_batch_id,
                graph_execution_state_id=context.graph_execution_state_id,
                model_key=model_key,
                submodel=submodel,
            )

    @property
    def logger(self):
        return self._loader.logger

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
        merger = ModelMerger(self._loader.store)
        try:
            if not merged_model_name:
                merged_model_name = "+".join([self._loader.store.get_model(x).name for x in model_keys])
                raise Exception("not implemented")

            self.logger.error("ModelMerger needs to be rewritten.")
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

    def search_for_models(self, directory: Path) -> List[Path]:
        """
        Return list of all models found in the designated directory.

        :param directory: Path to the directory to recursively search.
        returns a list of model paths
        """
        return ModelSearch().search(directory)

    def sync_to_config(self):
        """
        Synchronize the model manager to the database.

        Re-read models.yaml, rescan the models directory, and reimport models
        in the autoimport directories. Call after making changes outside the
        model manager API.
        """
        return self._loader.sync_to_config()

    def list_checkpoint_configs(self) -> List[Path]:
        """List the checkpoint config paths from ROOT/configs/stable-diffusion."""
        config = self._loader.config
        conf_path = config.legacy_conf_path
        root_path = config.root_path
        return [(conf_path / x).relative_to(root_path) for x in conf_path.glob("**/*.yaml")]

    def rename_model(
        self,
        key: str,
        new_name: str,
    ):
        """
        Rename the indicated model to the new name.

        :param key: Unique key for the model.
        :param new_name: New name for the model
        """
        return self.update_model(key, {"name": new_name})
