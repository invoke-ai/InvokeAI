from typing import Union

from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.lora.layers.full_layer import FullLayer
from invokeai.backend.lora.layers.ia3_layer import IA3Layer
from invokeai.backend.lora.layers.loha_layer import LoHALayer
from invokeai.backend.lora.layers.lokr_layer import LoKRLayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.layers.norm_layer import NormLayer

AnyLoRALayer = Union[LoRALayer, LoHALayer, LoKRLayer, FullLayer, IA3Layer, NormLayer, ConcatenatedLoRALayer]
