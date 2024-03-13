"""
invokeai.backend.model_manager.merge exports:
merge_diffusion_models() -- combine multiple models by location and return a pipeline object
merge_diffusion_models_and_commit() -- combine multiple models by ModelManager ID and write to the models tables

Copyright (c) 2023 Lincoln Stein and the InvokeAI Development Team
"""

import warnings
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Set

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import logging as dlogging

from invokeai.app.services.model_install import ModelInstallServiceBase
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.backend.util.devices import choose_torch_device, torch_dtype

from . import (
    AnyModelConfig,
    BaseModelType,
    ModelType,
    ModelVariantType,
)
from .config import MainDiffusersConfig


class MergeInterpolationMethod(str, Enum):
    WeightedSum = "weighted_sum"
    Sigmoid = "sigmoid"
    InvSigmoid = "inv_sigmoid"
    AddDifference = "add_difference"


class ModelMerger(object):
    """Wrapper class for model merge function."""

    def __init__(self, installer: ModelInstallServiceBase):
        """
        Initialize a ModelMerger object with the model installer.
        """
        self._installer = installer

    def merge_diffusion_models(
        self,
        model_paths: List[Path],
        alpha: float = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: bool = False,
        variant: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:  # pipe.merge is an untyped function.
        """
        :param model_paths:  up to three models, designated by their local paths or HuggingFace repo_ids
        :param alpha: The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
                   would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2
        :param interp: The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_difference" and None.
                   Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_difference" is supported.
        :param force:  Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

        **kwargs - the default DiffusionPipeline.get_config_dict kwargs:
             cache_dir, resume_download, force_download, proxies, local_files_only, use_auth_token, revision, torch_dtype, device_map
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verbosity = dlogging.get_verbosity()
            dlogging.set_verbosity_error()
            dtype = torch.float16 if variant == "fp16" else torch_dtype(choose_torch_device())

            # Note that checkpoint_merger will not work with downloaded HuggingFace fp16 models
            # until upstream https://github.com/huggingface/diffusers/pull/6670 is merged and released.
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_paths[0],
                custom_pipeline="checkpoint_merger",
                torch_dtype=dtype,
                variant=variant,
            )  # type: ignore
            merged_pipe = pipe.merge(
                pretrained_model_name_or_path_list=model_paths,
                alpha=alpha,
                interp=interp.value if interp else None,  # diffusers API treats None as "weighted sum"
                force=force,
                torch_dtype=dtype,
                variant=variant,
                **kwargs,
            )
            dlogging.set_verbosity(verbosity)
        return merged_pipe

    def merge_diffusion_models_and_save(
        self,
        model_keys: List[str],
        merged_model_name: str,
        alpha: float = 0.5,
        force: bool = False,
        interp: Optional[MergeInterpolationMethod] = None,
        merge_dest_directory: Optional[Path] = None,
        variant: Optional[str] = None,
        **kwargs: Any,
    ) -> AnyModelConfig:
        """
        :param models: up to three models, designated by their registered InvokeAI model name
        :param merged_model_name: name for new model
        :param alpha: The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
                   would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2
        :param interp: The interpolation method to use for the merging. Supports "weighted_average", "sigmoid", "inv_sigmoid", "add_difference" and None.
                   Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_difference" is supported. Add_difference is A+(B-C).
        :param force:  Whether to ignore mismatch in model_config.json for the current models. Defaults to False.
        :param merge_dest_directory: Save the merged model to the designated directory (with 'merged_model_name' appended)
        **kwargs - the default DiffusionPipeline.get_config_dict kwargs:
             cache_dir, resume_download, force_download, proxies, local_files_only, use_auth_token, revision, torch_dtype, device_map
        """
        model_paths: List[Path] = []
        model_names: List[str] = []
        config = self._installer.app_config
        store = self._installer.record_store
        base_models: Set[BaseModelType] = set()
        variant = None if self._installer.app_config.full_precision else "fp16"

        assert (
            len(model_keys) <= 2 or interp == MergeInterpolationMethod.AddDifference
        ), "When merging three models, only the 'add_difference' merge method is supported"

        for key in model_keys:
            info = store.get_model(key)
            model_names.append(info.name)
            assert isinstance(
                info, MainDiffusersConfig
            ), f"{info.name} ({info.key}) is not a diffusers model. It must be optimized before merging"
            assert info.variant == ModelVariantType(
                "normal"
            ), f"{info.name} ({info.key}) is a {info.variant} model, which cannot currently be merged"

            # tally base models used
            base_models.add(info.base)
            model_paths.extend([config.models_path / info.path])

        assert len(base_models) == 1, f"All models to merge must have same base model, but found bases {base_models}"
        base_model = base_models.pop()

        merge_method = None if interp == "weighted_sum" else MergeInterpolationMethod(interp)
        merged_pipe = self.merge_diffusion_models(model_paths, alpha, merge_method, force, variant=variant, **kwargs)
        dump_path = (
            Path(merge_dest_directory)
            if merge_dest_directory
            else config.models_path / base_model.value / ModelType.Main.value
        )
        dump_path.mkdir(parents=True, exist_ok=True)
        dump_path = dump_path / merged_model_name

        dtype = torch.float16 if variant == "fp16" else torch_dtype(choose_torch_device())
        merged_pipe.save_pretrained(dump_path.as_posix(), safe_serialization=True, torch_dtype=dtype, variant=variant)

        # register model and get its unique key
        key = self._installer.register_path(dump_path)

        # update model's config
        model_config = self._installer.record_store.get_model(key)
        model_config.name = merged_model_name
        model_config.description = f"Merge of models {', '.join(model_names)}"

        self._installer.record_store.update_model(
            key, ModelRecordChanges(name=model_config.name, description=model_config.description)
        )
        return model_config
