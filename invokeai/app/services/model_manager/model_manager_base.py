# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

from pydantic import Field

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.invocation_context import InvocationContextData
from invokeai.backend.model_management import (
    AddModelResult,
    BaseModelType,
    LoadedModelInfo,
    MergeInterpolationMethod,
    ModelType,
    SchedulerPredictionType,
    SubModelType,
)
from invokeai.backend.model_management.model_cache import CacheStats


class ModelManagerServiceBase(ABC):
    """Responsible for managing models on disk and in memory"""

    @abstractmethod
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
        pass

    @abstractmethod
    def get_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: Optional[SubModelType] = None,
        context_data: Optional[InvocationContextData] = None,
    ) -> LoadedModelInfo:
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
            default=None, min_length=2, max_length=3, description="List of model names to merge"
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
