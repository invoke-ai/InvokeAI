from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.ia3_layer import IA3Layer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.norm_layer import NormLayer
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)


def _without_unsupported_magnitude(state_dict: Dict[str, torch.Tensor], layer_type: str) -> Dict[str, torch.Tensor]:
    magnitude_keys = {"dora_scale", "dora_magnitude"} & state_dict.keys()
    if not magnitude_keys:
        return state_dict

    logger.warning(
        f"{layer_type} layer combines factorized weights with a DoRA magnitude, which is not supported; "
        f"applying it as plain {layer_type}."
    )
    return {key: value for key, value in state_dict.items() if key not in magnitude_keys}


def any_lora_layer_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> BaseLayerPatch:
    # Detect layers according to LyCORIS detection logic(`weight_list_det`)
    # https://github.com/KohakuBlueleaf/LyCORIS/tree/8ad8000efb79e2b879054da8c9356e6143591bad/lycoris/modules
    if "hada_w1_a" in state_dict:
        return LoHALayer.from_state_dict_values(_without_unsupported_magnitude(state_dict, "LoHa"))
    elif "lokr_w1" in state_dict or "lokr_w1_a" in state_dict:
        return LoKRLayer.from_state_dict_values(_without_unsupported_magnitude(state_dict, "LoKr"))
    elif "dora_scale" in state_dict or "dora_magnitude" in state_dict:
        # ``dora_scale`` is the LyCORIS magnitude, on either axis; ``dora_magnitude`` the PEFT/ai-toolkit out-dim one.
        return DoRALayer.from_state_dict_values(state_dict)
    elif "lora_up.weight" in state_dict:
        # LoRA a.k.a LoCon
        return LoRALayer.from_state_dict_values(state_dict)
    elif "diff" in state_dict:
        # Full a.k.a Diff
        return FullLayer.from_state_dict_values(state_dict)
    elif "on_input" in state_dict:
        return IA3Layer.from_state_dict_values(state_dict)
    elif "w_norm" in state_dict:
        return NormLayer.from_state_dict_values(state_dict)
    else:
        raise ValueError(f"Unsupported lora format: {state_dict.keys()}")
