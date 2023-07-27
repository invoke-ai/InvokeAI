"""
invokeai.backend.model_management.model_merge exports:
merge_diffusion_models() -- combine multiple models by location and return a pipeline object
merge_diffusion_models_and_commit() -- combine multiple models by ModelManager ID and write to models.yaml

Copyright (c) 2023 Lincoln Stein and the InvokeAI Development Team
"""

import warnings
from enum import Enum
from pathlib import Path
from diffusers import DiffusionPipeline
from diffusers import logging as dlogging
from typing import List, Union, Optional

import invokeai.backend.util.logging as logger

from ...backend.model_management import ModelManager, ModelType, BaseModelType, ModelVariantType, AddModelResult


class MergeInterpolationMethod(str, Enum):
    WeightedSum = "weighted_sum"
    Sigmoid = "sigmoid"
    InvSigmoid = "inv_sigmoid"
    AddDifference = "add_difference"


class ModelMerger(object):
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def merge_diffusion_models(
        self,
        model_paths: List[Path],
        alpha: float = 0.5,
        interp: MergeInterpolationMethod = None,
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
        model_names: List[str],
        base_model: Union[BaseModelType, str],
        merged_model_name: str,
        alpha: float = 0.5,
        interp: MergeInterpolationMethod = None,
        force: bool = False,
        merge_dest_directory: Optional[Path] = None,
        **kwargs,
    ) -> AddModelResult:
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
        model_paths = list()
        config = self.manager.app_config
        base_model = BaseModelType(base_model)
        vae = None

        for mod in model_names:
            info = self.manager.list_model(mod, base_model=base_model, model_type=ModelType.Main)
            assert info, f"model {mod}, base_model {base_model}, is unknown"
            assert (
                info["model_format"] == "diffusers"
            ), f"{mod} is not a diffusers model. It must be optimized before merging"
            assert info["variant"] == "normal", f"{mod} is a {info['variant']} model, which cannot currently be merged"
            assert (
                len(model_names) <= 2 or interp == MergeInterpolationMethod.AddDifference
            ), "When merging three models, only the 'add_difference' merge method is supported"
            # pick up the first model's vae
            if mod == model_names[0]:
                vae = info.get("vae")
            model_paths.extend([config.root_path / info["path"]])

        merge_method = None if interp == "weighted_sum" else MergeInterpolationMethod(interp)
        logger.debug(f"interp = {interp}, merge_method={merge_method}")
        merged_pipe = self.merge_diffusion_models(model_paths, alpha, merge_method, force, **kwargs)
        dump_path = (
            Path(merge_dest_directory)
            if merge_dest_directory
            else config.models_path / base_model.value / ModelType.Main.value
        )
        dump_path.mkdir(parents=True, exist_ok=True)
        dump_path = dump_path / merged_model_name

        merged_pipe.save_pretrained(dump_path, safe_serialization=1)
        attributes = dict(
            path=str(dump_path),
            description=f"Merge of models {', '.join(model_names)}",
            model_format="diffusers",
            variant=ModelVariantType.Normal.value,
            vae=vae,
        )
        return self.manager.add_model(
            merged_model_name,
            base_model=base_model,
            model_type=ModelType.Main,
            model_attributes=attributes,
            clobber=True,
        )
