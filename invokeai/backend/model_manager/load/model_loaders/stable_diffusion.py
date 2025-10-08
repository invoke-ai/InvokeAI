# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for StableDiffusion model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline,
)

from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    Checkpoint_Config_Base,
    Diffusers_Config_Base,
    Main_Checkpoint_SD1_Config,
    Main_Checkpoint_SD2_Config,
    Main_Checkpoint_SDXL_Config,
    Main_Checkpoint_SDXLRefiner_Config,
    Main_Diffusers_SD1_Config,
    Main_Diffusers_SD2_Config,
    Main_Diffusers_SDXL_Config,
    Main_Diffusers_SDXLRefiner_Config,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import get_model_cache_key
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    ModelVariantType,
    SubModelType,
)
from invokeai.backend.util.silence_warnings import SilenceWarnings

VARIANT_TO_IN_CHANNEL_MAP = {
    ModelVariantType.Normal: 4,
    ModelVariantType.Depth: 5,
    ModelVariantType.Inpaint: 9,
}


@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion1, type=ModelType.Main, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion2, type=ModelType.Main, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusionXL, type=ModelType.Main, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusionXLRefiner, type=ModelType.Main, format=ModelFormat.Diffusers
)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion3, type=ModelType.Main, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion1, type=ModelType.Main, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusion2, type=ModelType.Main, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusionXL, type=ModelType.Main, format=ModelFormat.Checkpoint)
@ModelLoaderRegistry.register(
    base=BaseModelType.StableDiffusionXLRefiner, type=ModelType.Main, format=ModelFormat.Checkpoint
)
class StableDiffusionDiffusersModel(GenericDiffusersLoader):
    """Class to load main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            return self._load_from_singlefile(config, submodel_type)

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
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
            BaseModelType.StableDiffusionXLRefiner: {
                ModelVariantType.Normal: StableDiffusionXLPipeline,
            },
        }
        assert isinstance(
            config,
            (
                Main_Diffusers_SD1_Config,
                Main_Diffusers_SD2_Config,
                Main_Diffusers_SDXL_Config,
                Main_Diffusers_SDXLRefiner_Config,
                Main_Checkpoint_SD1_Config,
                Main_Checkpoint_SD2_Config,
                Main_Checkpoint_SDXL_Config,
                Main_Checkpoint_SDXLRefiner_Config,
            ),
        )
        try:
            load_class = load_classes[config.base][config.variant]
        except KeyError as e:
            raise Exception(f"No diffusers pipeline known for base={config.base}, variant={config.variant}") from e

        # Without SilenceWarnings we get log messages like this:
        # site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
        # warnings.warn(
        # Some weights of the model checkpoint were not used when initializing CLIPTextModel:
        # ['text_model.embeddings.position_ids']
        # Some weights of the model checkpoint were not used when initializing CLIPTextModelWithProjection:
        # ['text_model.embeddings.position_ids']

        with SilenceWarnings():
            pipeline = load_class.from_single_file(config.path, torch_dtype=self._torch_dtype)

        if not submodel_type:
            return pipeline

        # Proactively load the various submodels into the RAM cache so that we don't have to re-load
        # the entire pipeline every time a new submodel is needed.
        for subtype in SubModelType:
            if subtype == submodel_type:
                continue
            if submodel := getattr(pipeline, subtype.value, None):
                self._ram_cache.put(get_model_cache_key(config.key, subtype), model=submodel)
        return getattr(pipeline, submodel_type.value)
