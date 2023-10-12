# Copyright 2023 Lincoln Stein and the InvokeAI Team

"""
Convert and merge models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from shutil import move, rmtree
from typing import List, Optional

from pydantic import Field

from invokeai.backend.model_manager.merge import MergeInterpolationMethod, ModelMerger

from .config import InvokeAIAppConfig
from .model_install_service import ModelInstallServiceBase
from .model_loader_service import ModelInfo, ModelLoadServiceBase
from .model_record_service import ModelConfigBase, ModelRecordServiceBase, ModelType, SubModelType


class ModelConvertBase(ABC):
    """Convert and merge models."""

    @abstractmethod
    def __init__(
        cls,
        loader: ModelLoadServiceBase,
        installer: ModelInstallServiceBase,
        store: ModelRecordServiceBase,
    ):
        """Initialize ModelConvert with loader, installer and configuration store."""
        pass

    @abstractmethod
    def convert_model(
        self,
        key: str,
        dest_directory: Optional[Path] = None,
    ) -> ModelConfigBase:
        """
        Convert a checkpoint file into a diffusers folder.

        It will delete the cached version ans well as the
        original checkpoint file if it is in the models directory.
        :param key: Unique key of model.
        :dest_directory: Optional place to put converted file. If not specified,
        will be stored in the `models_dir`.

        This will raise a ValueError unless the model is a checkpoint.
        This will raise an UnknownModelException if key is unknown.
        """
        pass

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
        pass


class ModelConvert(ModelConvertBase):
    """Implementation of ModelConvertBase."""

    def __init__(
        self,
        loader: ModelLoadServiceBase,
        installer: ModelInstallServiceBase,
        store: ModelRecordServiceBase,
    ):
        """Initialize ModelConvert with loader, installer and configuration store."""
        self.loader = loader
        self.installer = installer
        self.store = store

    def convert_model(
        self,
        key: str,
        dest_directory: Optional[Path] = None,
    ) -> ModelConfigBase:
        """
        Convert a checkpoint file into a diffusers folder.

        It will delete the cached version as well as the
        original checkpoint file if it is in the models directory.
        :param key: Unique key of model.
        :dest_directory: Optional place to put converted file. If not specified,
        will be stored in the `models_dir`.

        This will raise a ValueError unless the model is a checkpoint.
        This will raise an UnknownModelException if key is unknown.
        """
        new_diffusers_path = None
        config = InvokeAIAppConfig.get_config()

        try:
            info: ModelConfigBase = self.store.get_model(key)

            if info.model_format != "checkpoint":
                raise ValueError(f"not a checkpoint format model: {info.name}")

            # We are taking advantage of a side effect of get_model() that converts check points
            # into cached diffusers directories stored at `path`. It doesn't matter
            # what submodel type we request here, so we get the smallest.
            submodel = {"submodel_type": SubModelType.Scheduler} if info.model_type == ModelType.Main else {}
            converted_model: ModelInfo = self.loader.get_model(key, **submodel)

            checkpoint_path = config.models_path / info.path
            old_diffusers_path = config.models_path / converted_model.location

            # new values to write in
            update = info.dict()
            update.pop("config")
            update["model_format"] = "diffusers"
            update["path"] = str(converted_model.location)

            if dest_directory:
                new_diffusers_path = Path(dest_directory) / info.name
                if new_diffusers_path.exists():
                    raise ValueError(f"A diffusers model already exists at {new_diffusers_path}")
                move(old_diffusers_path, new_diffusers_path)
                update["path"] = new_diffusers_path.as_posix()

            self.store.update_model(key, update)
            result = self.installer.sync_model_path(key, ignore_hash_change=True)
        except Exception as excp:
            # something went wrong, so don't leave dangling diffusers model in directory or it will cause a duplicate model error!
            if new_diffusers_path:
                rmtree(new_diffusers_path)
            raise excp

        if checkpoint_path.exists() and checkpoint_path.is_relative_to(config.models_path):
            checkpoint_path.unlink()

        return result

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
        pass
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
