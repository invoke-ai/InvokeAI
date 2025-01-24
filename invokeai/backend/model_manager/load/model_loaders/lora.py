# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for LoRA model loading in InvokeAI."""

from logging import Logger
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import (
    is_state_dict_likely_flux_control,
    lora_model_from_flux_control_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    lora_model_from_flux_diffusers_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    is_state_dict_likely_in_flux_kohya_format,
    lora_model_from_flux_kohya_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_onetrainer_lora_conversion_utils import (
    is_state_dict_likely_in_flux_onetrainer_format,
    lora_model_from_flux_onetrainer_state_dict,
)
from invokeai.backend.patches.lora_conversions.sd_lora_conversion_utils import lora_model_from_sd_state_dict
from invokeai.backend.patches.lora_conversions.sdxl_lora_conversion_utils import convert_sdxl_keys_to_diffusers_format


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.LoRA, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.LoRA, format=ModelFormat.LyCORIS)
@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.ControlLoRa, format=ModelFormat.LyCORIS)
@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.ControlLoRa, format=ModelFormat.Diffusers)
class LoRALoader(ModelLoader):
    """Class to load LoRA models."""

    # We cheat a little bit to get access to the model base
    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCache,
    ):
        """Initialize the loader."""
        super().__init__(app_config, logger, ram_cache)
        self._model_base: Optional[BaseModelType] = None

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("There are no submodels in a LoRA model.")
        model_path = Path(config.path)
        assert self._model_base is not None

        # Load the state dict from the model file.
        if model_path.suffix == ".safetensors":
            state_dict = load_file(model_path.absolute().as_posix(), device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu")

        # Apply state_dict key conversions, if necessary.
        if self._model_base == BaseModelType.StableDiffusionXL:
            state_dict = convert_sdxl_keys_to_diffusers_format(state_dict)
            model = lora_model_from_sd_state_dict(state_dict=state_dict)
        elif self._model_base == BaseModelType.Flux:
            if config.format == ModelFormat.Diffusers:
                # HACK(ryand): We set alpha=None for diffusers PEFT format models. These models are typically
                # distributed as a single file without the associated metadata containing the alpha value. We chose
                # alpha=None, because this is treated as alpha=rank internally in `LoRALayerBase.scale()`. alpha=rank
                # is a popular choice. For example, in the diffusers training scripts:
                # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py#L1194
                model = lora_model_from_flux_diffusers_state_dict(state_dict=state_dict, alpha=None)
            elif config.format == ModelFormat.LyCORIS:
                if is_state_dict_likely_in_flux_kohya_format(state_dict=state_dict):
                    model = lora_model_from_flux_kohya_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_in_flux_onetrainer_format(state_dict=state_dict):
                    model = lora_model_from_flux_onetrainer_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_flux_control(state_dict=state_dict):
                    model = lora_model_from_flux_control_state_dict(state_dict=state_dict)
                else:
                    raise ValueError(f"LoRA model is in unsupported FLUX format: {config.format}")
            else:
                raise ValueError(f"LoRA model is in unsupported FLUX format: {config.format}")
        elif self._model_base in [BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2]:
            # Currently, we don't apply any conversions for SD1 and SD2 LoRA models.
            model = lora_model_from_sd_state_dict(state_dict=state_dict)
        else:
            raise ValueError(f"Unsupported LoRA base model: {self._model_base}")

        model.to(dtype=self._torch_dtype)
        return model

    def _get_model_path(self, config: AnyModelConfig) -> Path:
        # cheating a little - we remember this variable for using in the subsequent call to _load_model()
        self._model_base = config.base

        model_base_path = self._app_config.models_path
        model_path = model_base_path / config.path

        if config.format == ModelFormat.Diffusers:
            for ext in ["safetensors", "bin"]:  # return path to the safetensors file inside the folder
                path = model_base_path / config.path / f"pytorch_lora_weights.{ext}"
                if path.exists():
                    model_path = path
                    break

        return model_path.resolve()
