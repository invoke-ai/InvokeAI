# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for StableDiffusion model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    SchedulerPredictionType,
    SubModelType,
)
from invokeai.backend.model_manager.config import CheckpointConfigBase, MainCheckpointConfig
from invokeai.backend.model_manager.convert_ckpt_to_diffusers import convert_ckpt_to_diffusers

from .. import ModelLoaderRegistry
from .generic_diffusers import GenericDiffusersLoader


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
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not submodel_type is not None:
            raise Exception("A submodel type must be provided when loading main pipelines.")
        load_class = self.get_hf_load_class(model_path, submodel_type)
        variant = model_variant.value if model_variant else None
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

    def _needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool:
        if not isinstance(config, CheckpointConfigBase):
            return False
        elif (
            dest_path.exists()
            and (dest_path / "model_index.json").stat().st_mtime >= (config.converted_at or 0.0)
            and (dest_path / "model_index.json").stat().st_mtime >= model_path.stat().st_mtime
        ):
            return False
        else:
            return True

    def _convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Path) -> Path:
        assert isinstance(config, MainCheckpointConfig)
        base = config.base

        prediction_type = config.prediction_type.value
        upcast_attention = config.upcast_attention
        image_size = (
            1024
            if base == BaseModelType.StableDiffusionXL
            else 768
            if config.prediction_type == SchedulerPredictionType.VPrediction and base == BaseModelType.StableDiffusion2
            else 512
        )

        self._logger.info(f"Converting {model_path} to diffusers format")
        convert_ckpt_to_diffusers(
            model_path,
            output_path,
            model_type=self.model_base_to_model_type[base],
            original_config_file=self._app_config.root_path / config.config_path,
            extract_ema=True,
            from_safetensors=model_path.suffix == ".safetensors",
            precision=self._torch_dtype,
            prediction_type=prediction_type,
            image_size=image_size,
            upcast_attention=upcast_attention,
            load_safety_checker=False,
        )
        return output_path
