"""
invokeai.backend.model_manager.merge exports:
merge_diffusion_models() -- combine multiple models by location and return a pipeline object
merge_diffusion_models_and_commit() -- combine multiple models by ModelManager ID and write to models.yaml

Copyright (c) 2023 Lincoln Stein and the InvokeAI Development Team
"""

import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

from diffusers import DiffusionPipeline
from diffusers import logging as dlogging

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig

from . import BaseModelType, ModelConfigBase, ModelConfigStore, ModelType
from .loader import ModelLoad
from .config import MainConfig


class MergeInterpolationMethod(str, Enum):
    WeightedSum = "weighted_sum"
    Sigmoid = "sigmoid"
    InvSigmoid = "inv_sigmoid"
    AddDifference = "add_difference"


class ModelMerger(object):
    _store: ModelConfigStore
    _config: InvokeAIAppConfig

    def __init__(self, store: ModelConfigStore, config: Optional[InvokeAIAppConfig] = None):
        """
        Initialize a ModelMerger object.

        :param store: Underlying storage manager for the running process.
        :param config: InvokeAIAppConfig object (if not provided, default will be selected).
        """
        self._store = store
        self._config = config or InvokeAIAppConfig.get_config()

    def merge_diffusion_models(
        self,
        model_paths: List[Path],
        alpha: float = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: bool = False,
        **kwargs,
    ) -> DiffusionPipeline:
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

            pipe = DiffusionPipeline.from_pretrained(
                model_paths[0],
                custom_pipeline="checkpoint_merger",
            )
            merged_pipe = pipe.merge(
                pretrained_model_name_or_path_list=model_paths,
                alpha=alpha,
                interp=interp.value if interp else None,  # diffusers API treats None as "weighted sum"
                force=force,
                **kwargs,
            )
            dlogging.set_verbosity(verbosity)
        return merged_pipe

    def merge_diffusion_models_and_save(
        self,
        model_keys: List[str],
        merged_model_name: str,
        alpha: Optional[float] = 0.5,
        interp: Optional[MergeInterpolationMethod] = None,
        force: Optional[bool] = False,
        merge_dest_directory: Optional[Path] = None,
        **kwargs,
    ) -> ModelConfigBase:
        """
        :param models: up to three models, designated by their InvokeAI models.yaml model name
        :param base_model: base model (must be the same for all merged models!)
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
        model_paths: List[Path] = list()
        model_names = list()
        config = self._config
        store = self._store
        base_models: Set[BaseModelType] = set()
        vae = None

        assert (
            len(model_keys) <= 2 or interp == MergeInterpolationMethod.AddDifference
        ), "When merging three models, only the 'add_difference' merge method is supported"

        for key in model_keys:
            info = store.get_model(key)
            assert isinstance(info, MainConfig)
            model_names.append(info.name)
            assert (
                info.model_format == "diffusers"
            ), f"{info.name} ({info.key}) is not a diffusers model. It must be optimized before merging"
            assert (
                info.variant == "normal"
            ), f"{info.name} ({info.key}) is a {info.variant} model, which cannot currently be merged"

            # pick up the first model's vae
            if key == model_keys[0]:
                vae = info.vae

            # tally base models used
            base_models.add(info.base_model)
            model_paths.extend([(config.models_path / info.path).as_posix()])

        assert len(base_models) == 1, f"All models to merge must have same base model, but found bases {base_models}"
        base_model = base_models.pop()

        merge_method = None if interp == "weighted_sum" else MergeInterpolationMethod(interp)
        logger.debug(f"interp = {interp}, merge_method={merge_method}")
        merged_pipe = self.merge_diffusion_models(model_paths, alpha, merge_method, force, **kwargs)
        dump_path = (
            Path(merge_dest_directory)
            if merge_dest_directory
            else config.models_path / base_model.value / ModelType.Main.value
        )
        dump_path.mkdir(parents=True, exist_ok=True)
        dump_path = (dump_path / merged_model_name).as_posix()

        merged_pipe.save_pretrained(dump_path, safe_serialization=True)

        # register model and get its unique key
        installer = ModelInstall(store=self._store, config=self._config)
        key = installer.register_path(dump_path)

        # update model's config
        model_config = self._store.get_model(key)
        model_config.update(
            dict(
                name=merged_model_name,
                description=f"Merge of models {', '.join(model_names)}",
                vae=vae,
            )
        )
        self._store.update_model(key, model_config)
        return model_config
