from typing import Union

from invokeai.backend.patches.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.ia3_layer import IA3Layer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.norm_layer import NormLayer
from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer

AnyLoRALayer = Union[
    LoRALayer, LoHALayer, LoKRLayer, FullLayer, IA3Layer, NormLayer, ConcatenatedLoRALayer, SetParameterLayer
]
