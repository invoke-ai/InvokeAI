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


def any_lora_layer_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> BaseLayerPatch:
    # Detect layers according to LyCORIS detection logic(`weight_list_det`)
    # https://github.com/KohakuBlueleaf/LyCORIS/tree/8ad8000efb79e2b879054da8c9356e6143591bad/lycoris/modules
    if "dora_scale" in state_dict:
        return DoRALayer.from_state_dict_values(state_dict)
    elif "lora_up.weight" in state_dict:
        # LoRA a.k.a LoCon
        return LoRALayer.from_state_dict_values(state_dict)
    elif "hada_w1_a" in state_dict:
        return LoHALayer.from_state_dict_values(state_dict)
    elif "lokr_w1" in state_dict or "lokr_w1_a" in state_dict:
        return LoKRLayer.from_state_dict_values(state_dict)
    elif "diff" in state_dict:
        # Full a.k.a Diff
        return FullLayer.from_state_dict_values(state_dict)
    elif "on_input" in state_dict:
        return IA3Layer.from_state_dict_values(state_dict)
    elif "w_norm" in state_dict:
        return NormLayer.from_state_dict_values(state_dict)
    else:
        raise ValueError(f"Unsupported lora format: {state_dict.keys()}")
