import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
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
    transformer: dict = field(default_factory=dict)
    # might also grow CLIP and T5 submodels


def _group_state_by_submodel(state_dict: dict[str, torch.Tensor]) -> GroupedStateDict:
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


def lora_model_from_flux_aitoolkit_state_dict(state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    grouped = _group_state_by_submodel(state_dict)

    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, layer_state_dict in grouped.transformer.items():
        layers[FLUX_LORA_TRANSFORMER_PREFIX + layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    return ModelPatchRaw(layers=layers)
