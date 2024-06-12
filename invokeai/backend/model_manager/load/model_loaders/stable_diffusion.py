# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for StableDiffusion model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    ModelVariantType,
    SubModelType,
)
from invokeai.backend.model_manager.config import (
    CheckpointConfigBase,
    DiffusersConfigBase,
    MainCheckpointConfig,
)
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data

from .. import ModelLoaderRegistry
from .generic_diffusers import GenericDiffusersLoader

VARIANT_TO_IN_CHANNEL_MAP = {
    ModelVariantType.Normal: 4,
    ModelVariantType.Depth: 5,
    ModelVariantType.Inpaint: 9,
}


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Main, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Main, format=ModelFormat.Checkpoint)
class StableDiffusionDiffusersModel(GenericDiffusersLoader):
    """Class to load main models."""

    model_base_to_model_type = {
        BaseModelType.StableDiffusion1: "FrozenCLIPEmbedder",
        BaseModelType.StableDiffusion2: "FrozenOpenCLIPEmbedder",
        BaseModelType.StableDiffusionXL: "SDXL",
        BaseModelType.StableDiffusionXLRefiner: "SDXL-Refiner",
    }

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, CheckpointConfigBase):
            return self._load_from_singlefile(config, submodel_type)

        if not submodel_type is not None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, DiffusersConfigBase) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                torch_dtype=self._torch_dtype,
                variant=variant,
            )
        except OSError as e:
            if variant and "no file named" in str(
                e
            ):  # try without the variant, just in case user's preferences changed
                result = load_class.from_pretrained(model_path, torch_dtype=self._torch_dtype)
            else:
                raise e

        return result

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        load_classes = {
            BaseModelType.StableDiffusion1: {
                ModelVariantType.Normal: StableDiffusionPipeline,
                ModelVariantType.Inpaint: StableDiffusionInpaintPipeline,
            },
            BaseModelType.StableDiffusion2: {
                ModelVariantType.Normal: StableDiffusionPipeline,
                ModelVariantType.Inpaint: StableDiffusionInpaintPipeline,
            },
            BaseModelType.StableDiffusionXL: {
                ModelVariantType.Normal: StableDiffusionXLPipeline,
                ModelVariantType.Inpaint: StableDiffusionXLInpaintPipeline,
            },
        }
        assert isinstance(config, MainCheckpointConfig)
        try:
            load_class = load_classes[config.base][config.variant]
        except KeyError as e:
            raise Exception(f"No diffusers pipeline known for base={config.base}, variant={config.variant}") from e
        original_config_file = self._app_config.legacy_conf_path / config.config_path
        prediction_type = config.prediction_type.value
        upcast_attention = config.upcast_attention

        pipeline = load_class.from_single_file(
            config.path,
            config=original_config_file,
            torch_dtype=self._torch_dtype,
            local_files_only=True,
            prediction_type=prediction_type,
            upcast_attention=upcast_attention,
            load_safety_checker=False,
        )

        # Proactively load the various submodels into the RAM cache so that we don't have to re-load
        # the entire pipeline every time a new submodel is needed.
        if not submodel_type:
            return pipeline

        for subtype in SubModelType:
            if subtype == submodel_type:
                continue
            if submodel := getattr(pipeline, subtype.value, None):
                self._ram_cache.put(
                    config.key, submodel_type=subtype, model=submodel, size=calc_model_size_by_data(submodel)
                )
        return getattr(pipeline, submodel_type.value)
