import json
from dataclasses import dataclass, field
from typing import Any

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import _group_by_layer
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util import InvokeAILogger


def is_state_dict_likely_in_flux_aitoolkit_format(state_dict: dict[str, Any], metadata: dict[str, Any] = None) -> bool:
    if metadata:
        try:
            software = json.loads(metadata.get("software", "{}"))
        except json.JSONDecodeError:
            return False
        return software.get("name") == "ai-toolkit"
    # metadata got lost somewhere
    return any("diffusion_model" == k.split(".", 1)[0] for k in state_dict.keys())


@dataclass
class GroupedStateDict:
    transformer: dict[str, Any] = field(default_factory=dict)
    # might also grow CLIP and T5 submodels


def _group_state_by_submodel(state_dict: dict[str, Any]) -> GroupedStateDict:
    logger = InvokeAILogger.get_logger()
    grouped = GroupedStateDict()
    for key, value in state_dict.items():
        submodel_name, param_name = key.split(".", 1)
        match submodel_name:
            case "diffusion_model":
                grouped.transformer[param_name] = value
            case _:
                logger.warning(f"Unexpected submodel name: {submodel_name}")
    return grouped


def _rename_peft_lora_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Renames keys from the PEFT LoRA format to the InvokeAI format."""
    renamed_state_dict = {}
    for key, value in state_dict.items():
        renamed_key = key.replace(".lora_A.", ".lora_down.").replace(".lora_B.", ".lora_up.")
        renamed_state_dict[renamed_key] = value
    return renamed_state_dict


def lora_model_from_flux_aitoolkit_state_dict(state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    state_dict = _rename_peft_lora_keys(state_dict)
    by_layer = _group_by_layer(state_dict)
    by_model = _group_state_by_submodel(by_layer)

    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, layer_state_dict in by_model.transformer.items():
        layers[FLUX_LORA_TRANSFORMER_PREFIX + layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    return ModelPatchRaw(layers=layers)
