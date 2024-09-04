from typing import Dict

import torch

from invokeai.backend.peft.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.peft.layers.full_layer import FullLayer
from invokeai.backend.peft.layers.ia3_layer import IA3Layer
from invokeai.backend.peft.layers.loha_layer import LoHALayer
from invokeai.backend.peft.layers.lokr_layer import LoKRLayer
from invokeai.backend.peft.layers.lora_layer import LoRALayer
from invokeai.backend.peft.layers.norm_layer import NormLayer


def peft_layer_from_state_dict(layer_key: str, state_dict: Dict[str, torch.Tensor]) -> AnyLoRALayer:
    # Detect layers according to LyCORIS detection logic(`weight_list_det`)
    # https://github.com/KohakuBlueleaf/LyCORIS/tree/8ad8000efb79e2b879054da8c9356e6143591bad/lycoris/modules

    if "lora_up.weight" in state_dict:
        # LoRA a.k.a LoCon
        return LoRALayer(layer_key, state_dict)
    elif "hada_w1_a" in state_dict:
        return LoHALayer(layer_key, state_dict)
    elif "lokr_w1" in state_dict or "lokr_w1_a" in state_dict:
        return LoKRLayer(layer_key, state_dict)
    elif "diff" in state_dict:
        # Full a.k.a Diff
        return FullLayer(layer_key, state_dict)
    elif "on_input" in state_dict:
        return IA3Layer(layer_key, state_dict)
    elif "w_norm" in state_dict:
        return NormLayer(layer_key, state_dict)
    else:
        raise ValueError(f"Unsupported lora format: {state_dict.keys()}")
