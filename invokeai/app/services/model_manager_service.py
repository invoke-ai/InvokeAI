# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from pydantic import Field
from typing import Literal, Optional, Union, Callable, List, Tuple, TYPE_CHECKING
from types import ModuleType

from invokeai.backend.model_management import (
    ModelManager,
    BaseModelType,
    ModelType,
    SubModelType,
    ModelInfo,
    AddModelResult,
    SchedulerPredictionType,
    ModelMerger,
    MergeInterpolationMethod,
    ModelNotFoundException,
)
from invokeai.backend.model_management.model_search import FindModels
from invokeai.backend.model_management.model_cache import CacheStats

import torch
from invokeai.app.models.exceptions import CanceledException
from ...backend.util import choose_precision, choose_torch_device
from .config import InvokeAIAppConfig

if TYPE_CHECKING:
    from ..invocations.baseinvocation import BaseInvocation, InvocationContext


class ModelManagerServiceBase(ABC):
    """Responsible for managing models on disk and in memory"""

    @abstractmethod
    def __init__(
        self,
        config: InvokeAIAppConfig,
        logger: ModuleType,
    ):
        """
        Initialize with the path to the models.yaml config file.
        Optional parameters are the torch device type, precision, max_models,
        and sequential_offload boolean. Note that the default device
        type and precision are set up for a CUDA system running at half precision.
        """
        pass

    @abstractmethod
    def get_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        node: Optional[BaseInvocation] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """Retrieve the indicated model with name and type.
        submodel can be used to get a part (such as the vae)
        of a diffusers pipeline."""
        pass

    @property
    @abstractmethod
    def logger(self):
        pass

    @abstractmethod
    def model_exists(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> bool:
        pass

    @abstractmethod
    def model_info(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> dict:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        Uses the exact format as the omegaconf stanza.
        """
        pass

    @abstractmethod
    def list_models(self, base_model: Optional[BaseModelType] = None, model_type: Optional[ModelType] = None) -> dict:
        """
        Return a dict of models in the format:
        { model_type1:
          { model_name1: {'status': 'active'|'cached'|'not loaded',
                         'model_name' : name,
                         'model_type' : SDModelType,
                         'description': description,
                         'format': 'folder'|'safetensors'|'ckpt'
                         },
            model_name2: { etc }
          },
          model_type2:
            { model_name_n: etc
        }
        """
        pass

    @abstractmethod
    def list_model(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> dict:
        """
        Return information about the model using the same format as list_models()
        """
        pass

    @abstractmethod
    def model_names(self) -> List[Tuple[str, BaseModelType, ModelType]]:
        """
        Returns a list of all the model names known.
        """
        pass

    @abstractmethod
    def add_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
        clobber: bool = False,
    ) -> AddModelResult:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model name is missing. Call commit() to write changes to disk.
        """
        pass

    @abstractmethod
    def update_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
    ) -> AddModelResult:
        """
        Update the named model with a dictionary of attributes. Will fail with a
        ModelNotFoundException if the name does not already exist.

        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model name is missing. Call commit() to write changes to disk.
        """
        pass

    @abstractmethod
    def del_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        """
        Delete the named model from configuration. If delete_files is true,
        then the underlying weight file or diffusers directory will be deleted
        as well. Call commit() to write to disk.
        """
        pass

    @abstractmethod
    def rename_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        new_name: str,
    ):
        """
        Rename the indicated model.
        """
        pass

    @abstractmethod
    def list_checkpoint_configs(self) -> List[Path]:
        """
        List the checkpoint config paths from ROOT/configs/stable-diffusion.
        """
        pass

    @abstractmethod
    def convert_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: Literal[ModelType.Main, ModelType.Vae],
    ) -> AddModelResult:
        """
        Convert a checkpoint file into a diffusers folder, deleting the cached
        version and deleting the original checkpoint file if it is in the models
        directory.
        :param model_name: Name of the model to convert
        :param base_model: Base model type
        :param model_type: Type of model ['vae' or 'main']

        This will raise a ValueError unless the model is not a checkpoint. It will
        also raise a ValueError in the event that there is a similarly-named diffusers
        directory already in place.
        """
        pass

    @abstractmethod
    def heuristic_import(
        self,
        items_to_import: set[str],
        prediction_type_helper: Optional[Callable[[Path], SchedulerPredictionType]] = None,
    ) -> dict[str, AddModelResult]:
        """Import a list of paths, repo_ids or URLs. Returns the set of
        successfully imported items.
        :param items_to_import: Set of strings corresponding to models to be imported.
        :param prediction_type_helper: A callback that receives the Path of a Stable Diffusion 2 checkpoint model and returns a SchedulerPredictionType.

        The prediction type helper is necessary to distinguish between
        models based on Stable Diffusion 2 Base (requiring
        SchedulerPredictionType.Epsilson) and Stable Diffusion 768
        (requiring SchedulerPredictionType.VPrediction). It is
        generally impossible to do this programmatically, so the
        prediction_type_helper usually asks the user to choose.

        The result is a set of successfully installed models. Each element
        of the set is a dict corresponding to the newly-created OmegaConf stanza for
        that model.
        """
        pass

    @abstractmethod
    def merge_models(
        self,
        model_names: List[str] = Field(
            default=None, min_items=2, max_items=3, description="List of model names to merge"
        ),
        base_model: Union[BaseModelType, str] = Field(
            default=None, description="Base model shared by all models to be merged"
        ),
        merged_model_name: str = Field(default=None, description="Name of destination model after merging"),
        alpha: Optional[float] = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: Optional[bool] = False,
        merge_dest_directory: Optional[Path] = None,
    ) -> AddModelResult:
        """
        Merge two to three diffusrs pipeline models and save as a new model.
        :param model_names: List of 2-3 models to merge
        :param base_model: Base model to use for all models
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
        Re-read models.yaml, rescan the models directory, and reimport models
        in the autoimport directories. Call after making changes outside the
        model manager API.
        """
        pass

    @abstractmethod
    def collect_cache_stats(self, cache_stats: CacheStats):
        """
        Reset model cache statistics for graph with graph_id.
        """
        pass

    @abstractmethod
    def commit(self, conf_file: Optional[Path] = None) -> None:
        """
        Write current configuration out to the indicated file.
        If no conf_file is provided, then replaces the
        original file/database used to initialize the object.
        """
        pass


# simple implementation
class ModelManagerService(ModelManagerServiceBase):
    """Responsible for managing models on disk and in memory"""

    def __init__(
        self,
        config: InvokeAIAppConfig,
        logger: Logger,
    ):
        """
        Initialize with the path to the models.yaml config file.
        Optional parameters are the torch device type, precision, max_models,
        and sequential_offload boolean. Note that the default device
        type and precision are set up for a CUDA system running at half precision.
        """
        if config.model_conf_path and config.model_conf_path.exists():
            config_file = config.model_conf_path
        else:
            config_file = config.root_dir / "configs/models.yaml"

        logger.debug(f"Config file={config_file}")

        device = torch.device(choose_torch_device())
        device_name = torch.cuda.get_device_name() if device == torch.device("cuda") else ""
        logger.info(f"GPU device = {device} {device_name}")

        precision = config.precision
        if precision == "auto":
            precision = choose_precision(device)
        dtype = torch.float32 if precision == "float32" else torch.float16

        # this is transitional backward compatibility
        # support for the deprecated `max_loaded_models`
        # configuration value. If present, then the
        # cache size is set to 2.5 GB times
        # the number of max_loaded_models. Otherwise
        # use new `ram_cache_size` config setting
        max_cache_size = config.ram_cache_size

        logger.debug(f"Maximum RAM cache size: {max_cache_size} GiB")

        sequential_offload = config.sequential_guidance

        self.mgr = ModelManager(
            config=config_file,
            device_type=device,
            precision=dtype,
            max_cache_size=max_cache_size,
            sequential_offload=sequential_offload,
            logger=logger,
        )
        logger.info("Model manager service initialized")

    def get_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        context: Optional[InvocationContext] = None,
    ) -> ModelInfo:
        """
        Retrieve the indicated model. submodel can be used to get a
        part (such as the vae) of a diffusers mode.
        """

        # we can emit model loading events if we are executing with access to the invocation context
        if context:
            self._emit_load_event(
                context=context,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
            )

        model_info = self.mgr.get_model(
            model_name,
            base_model,
            model_type,
            submodel,
        )

        if context:
            self._emit_load_event(
                context=context,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
                model_info=model_info,
            )

        return model_info

    def model_exists(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        return self.mgr.model_exists(
            model_name,
            base_model,
            model_type,
        )

    def model_info(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> Union[dict, None]:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        """
        return self.mgr.model_info(model_name, base_model, model_type)

    def model_names(self) -> List[Tuple[str, BaseModelType, ModelType]]:
        """
        Returns a list of all the model names known.
        """
        return self.mgr.model_names()

    def list_models(
        self, base_model: Optional[BaseModelType] = None, model_type: Optional[ModelType] = None
    ) -> list[dict]:
        """
        Return a list of models.
        """
        return self.mgr.list_models(base_model, model_type)

    def list_model(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> Union[dict, None]:
        """
        Return information about the model using the same format as list_models()
        """
        return self.mgr.list_model(model_name=model_name, base_model=base_model, model_type=model_type)

    def add_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
        clobber: bool = False,
    ) -> AddModelResult:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model name is missing. Call commit() to write changes to disk.
        """
        self.logger.debug(f"add/update model {model_name}")
        return self.mgr.add_model(model_name, base_model, model_type, model_attributes, clobber)

    def update_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
    ) -> AddModelResult:
        """
        Update the named model with a dictionary of attributes. Will fail with a
        ModelNotFoundException exception if the name does not already exist.
        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model name is missing. Call commit() to write changes to disk.
        """
        self.logger.debug(f"update model {model_name}")
        if not self.model_exists(model_name, base_model, model_type):
            raise ModelNotFoundException(f"Unknown model {model_name}")
        return self.add_model(model_name, base_model, model_type, model_attributes, clobber=True)

    def del_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        """
        Delete the named model from configuration. If delete_files is true,
        then the underlying weight file or diffusers directory will be deleted
        as well.
        """
        self.logger.debug(f"delete model {model_name}")
        self.mgr.del_model(model_name, base_model, model_type)
        self.mgr.commit()

    def convert_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: Literal[ModelType.Main, ModelType.Vae],
        convert_dest_directory: Optional[Path] = Field(
            default=None, description="Optional directory location for merged model"
        ),
    ) -> AddModelResult:
        """
        Convert a checkpoint file into a diffusers folder, deleting the cached
        version and deleting the original checkpoint file if it is in the models
        directory.
        :param model_name: Name of the model to convert
        :param base_model: Base model type
        :param model_type: Type of model ['vae' or 'main']
        :param convert_dest_directory: Save the converted model to the designated directory (`models/etc/etc` by default)

        This will raise a ValueError unless the model is not a checkpoint. It will
        also raise a ValueError in the event that there is a similarly-named diffusers
        directory already in place.
        """
        self.logger.debug(f"convert model {model_name}")
        return self.mgr.convert_model(model_name, base_model, model_type, convert_dest_directory)

    def collect_cache_stats(self, cache_stats: CacheStats):
        """
        Reset model cache statistics for graph with graph_id.
        """
        self.mgr.cache.stats = cache_stats

    def commit(self, conf_file: Optional[Path] = None):
        """
        Write current configuration out to the indicated file.
        If no conf_file is provided, then replaces the
        original file/database used to initialize the object.
        """
        return self.mgr.commit(conf_file)

    def _emit_load_event(
        self,
        context,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        model_info: Optional[ModelInfo] = None,
    ):
        if context.services.queue.is_canceled(context.graph_execution_state_id):
            raise CanceledException()

        if model_info:
            context.services.events.emit_model_load_completed(
                graph_execution_state_id=context.graph_execution_state_id,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
                model_info=model_info,
            )
        else:
            context.services.events.emit_model_load_started(
                graph_execution_state_id=context.graph_execution_state_id,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
            )

    @property
    def logger(self):
        return self.mgr.logger

    def heuristic_import(
        self,
        items_to_import: set[str],
        prediction_type_helper: Optional[Callable[[Path], SchedulerPredictionType]] = None,
    ) -> dict[str, AddModelResult]:
        """Import a list of paths, repo_ids or URLs. Returns the set of
        successfully imported items.
        :param items_to_import: Set of strings corresponding to models to be imported.
        :param prediction_type_helper: A callback that receives the Path of a Stable Diffusion 2 checkpoint model and returns a SchedulerPredictionType.

        The prediction type helper is necessary to distinguish between
        models based on Stable Diffusion 2 Base (requiring
        SchedulerPredictionType.Epsilson) and Stable Diffusion 768
        (requiring SchedulerPredictionType.VPrediction). It is
        generally impossible to do this programmatically, so the
        prediction_type_helper usually asks the user to choose.

        The result is a set of successfully installed models. Each element
        of the set is a dict corresponding to the newly-created OmegaConf stanza for
        that model.
        """
        return self.mgr.heuristic_import(items_to_import, prediction_type_helper)

    def merge_models(
        self,
        model_names: List[str] = Field(
            default=None, min_items=2, max_items=3, description="List of model names to merge"
        ),
        base_model: Union[BaseModelType, str] = Field(
            default=None, description="Base model shared by all models to be merged"
        ),
        merged_model_name: str = Field(default=None, description="Name of destination model after merging"),
        alpha: float = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: bool = False,
        merge_dest_directory: Optional[Path] = Field(
            default=None, description="Optional directory location for merged model"
        ),
    ) -> AddModelResult:
        """
        Merge two to three diffusrs pipeline models and save as a new model.
        :param model_names: List of 2-3 models to merge
        :param base_model: Base model to use for all models
        :param merged_model_name: Name of destination merged model
        :param alpha: Alpha strength to apply to 2d and 3d model
        :param interp: Interpolation method. None (default)
        :param merge_dest_directory: Save the merged model to the designated directory (with 'merged_model_name' appended)
        """
        merger = ModelMerger(self.mgr)
        try:
            result = merger.merge_diffusion_models_and_save(
                model_names=model_names,
                base_model=base_model,
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
        """
        search = FindModels([directory], self.logger)
        return search.list_models()

    def sync_to_config(self):
        """
        Re-read models.yaml, rescan the models directory, and reimport models
        in the autoimport directories. Call after making changes outside the
        model manager API.
        """
        return self.mgr.sync_to_config()

    def list_checkpoint_configs(self) -> List[Path]:
        """
        List the checkpoint config paths from ROOT/configs/stable-diffusion.
        """
        config = self.mgr.app_config
        conf_path = config.legacy_conf_path
        root_path = config.root_path
        return [(conf_path / x).relative_to(root_path) for x in conf_path.glob("**/*.yaml")]

    def rename_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        new_name: Optional[str] = None,
        new_base: Optional[BaseModelType] = None,
    ):
        """
        Rename the indicated model. Can provide a new name and/or a new base.
        :param model_name: Current name of the model
        :param base_model: Current base of the model
        :param model_type: Model type (can't be changed)
        :param new_name: New name for the model
        :param new_base: New base for the model
        """
        self.mgr.rename_model(
            base_model=base_model,
            model_type=model_type,
            model_name=model_name,
            new_name=new_name,
            new_base=new_base,
        )
