# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for LoRA model loading in InvokeAI."""

from logging import Logger
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.omi.omi import convert_from_omi
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.patches.lora_conversions.flux_aitoolkit_lora_conversion_utils import (
    is_state_dict_likely_in_flux_aitoolkit_format,
    lora_model_from_flux_aitoolkit_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    is_state_dict_likely_in_flux_bfl_peft_format,
    lora_model_from_flux2_bfl_peft_state_dict,
    lora_model_from_flux_bfl_peft_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import (
    is_state_dict_likely_flux_control,
    lora_model_from_flux_control_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    is_state_dict_flux2_diffusers_format,
    is_state_dict_likely_in_flux_diffusers_format,
    lora_model_from_flux2_diffusers_state_dict,
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
from invokeai.backend.patches.lora_conversions.flux_xlabs_lora_conversion_utils import (
    is_state_dict_likely_in_flux_xlabs_format,
    lora_model_from_flux_xlabs_state_dict,
)
from invokeai.backend.patches.lora_conversions.sd_lora_conversion_utils import lora_model_from_sd_state_dict
from invokeai.backend.patches.lora_conversions.sdxl_lora_conversion_utils import convert_sdxl_keys_to_diffusers_format
from invokeai.backend.patches.lora_conversions.z_image_lora_conversion_utils import lora_model_from_z_image_state_dict


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.LoRA, format=ModelFormat.OMI)
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusionXL, type=ModelType.LoRA, format=ModelFormat.OMI)
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

        # Strip 'bundle_emb' keys - these are unused and currently cause downstream errors.
        # To revisit later to determine if they're needed/useful.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bundle_emb")}

        # At the time of writing, we support the OMI standard for base models Flux and SDXL
        if config.format == ModelFormat.OMI and self._model_base in [
            BaseModelType.StableDiffusionXL,
            BaseModelType.Flux,
        ]:
            state_dict = convert_from_omi(state_dict, config.base)  # type: ignore

        # Apply state_dict key conversions, if necessary.
        if self._model_base == BaseModelType.StableDiffusionXL:
            state_dict = convert_sdxl_keys_to_diffusers_format(state_dict)
            model = lora_model_from_sd_state_dict(state_dict=state_dict)
        elif self._model_base in (BaseModelType.Flux, BaseModelType.Flux2):
            if config.format is ModelFormat.OMI:
                # HACK(ryand): We set alpha=None for diffusers PEFT format models. These models are typically
                # distributed as a single file without the associated metadata containing the alpha value. We chose
                # alpha=None, because this is treated as alpha=rank internally in `LoRALayerBase.scale()`. alpha=rank
                # is a popular choice. For example, in the diffusers training scripts:
                # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py#L1194
                #
                # We assume the same for LyCORIS models in diffusers key format.
                model = lora_model_from_flux_diffusers_state_dict(state_dict=state_dict, alpha=None)
            elif config.format is ModelFormat.LyCORIS:
                if is_state_dict_likely_in_flux_diffusers_format(state_dict=state_dict):
                    if is_state_dict_flux2_diffusers_format(state_dict=state_dict):
                        # Flux2 Klein native diffusers naming (to_qkv_mlp_proj, ff.linear_in, etc.)
                        model = lora_model_from_flux2_diffusers_state_dict(state_dict=state_dict, alpha=None)
                    else:
                        # Flux.1 diffusers naming (to_q/to_k/to_v, ff.net.0.proj, etc.)
                        model = lora_model_from_flux_diffusers_state_dict(state_dict=state_dict, alpha=None)
                elif is_state_dict_likely_in_flux_kohya_format(state_dict=state_dict):
                    model = lora_model_from_flux_kohya_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_in_flux_onetrainer_format(state_dict=state_dict):
                    model = lora_model_from_flux_onetrainer_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_flux_control(state_dict=state_dict):
                    model = lora_model_from_flux_control_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_in_flux_aitoolkit_format(state_dict=state_dict):
                    model = lora_model_from_flux_aitoolkit_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_in_flux_xlabs_format(state_dict=state_dict):
                    model = lora_model_from_flux_xlabs_state_dict(state_dict=state_dict)
                elif is_state_dict_likely_in_flux_bfl_peft_format(state_dict=state_dict):
                    if self._model_base == BaseModelType.Flux2:
                        # FLUX.2 Klein uses Flux2Transformer2DModel (diffusers naming),
                        # so we need to convert BFL keys to diffusers naming.
                        model = lora_model_from_flux2_bfl_peft_state_dict(state_dict=state_dict, alpha=None)
                    else:
                        # FLUX.1 uses BFL Flux class, so BFL keys work directly.
                        model = lora_model_from_flux_bfl_peft_state_dict(state_dict=state_dict, alpha=None)
                else:
                    raise ValueError("LoRA model is in unsupported FLUX format")
            else:
                raise ValueError(f"LoRA model is in unsupported FLUX format: {config.format}")
        elif self._model_base in [BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2]:
            # Currently, we don't apply any conversions for SD1 and SD2 LoRA models.
            model = lora_model_from_sd_state_dict(state_dict=state_dict)
        elif self._model_base == BaseModelType.ZImage:
            # Z-Image LoRAs use diffusers PEFT format with transformer and/or Qwen3 encoder layers.
            # We set alpha=None to use rank as alpha (common default).
            model = lora_model_from_z_image_state_dict(state_dict=state_dict, alpha=None)
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
