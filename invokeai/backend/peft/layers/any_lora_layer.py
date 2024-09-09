from typing import Union

from invokeai.backend.peft.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.peft.layers.full_layer import FullLayer
from invokeai.backend.peft.layers.ia3_layer import IA3Layer
from invokeai.backend.peft.layers.loha_layer import LoHALayer
from invokeai.backend.peft.layers.lokr_layer import LoKRLayer
from invokeai.backend.peft.layers.lora_layer import LoRALayer
from invokeai.backend.peft.layers.norm_layer import NormLayer

AnyLoRALayer = Union[LoRALayer, LoHALayer, LoKRLayer, FullLayer, IA3Layer, NormLayer, ConcatenatedLoRALayer]
