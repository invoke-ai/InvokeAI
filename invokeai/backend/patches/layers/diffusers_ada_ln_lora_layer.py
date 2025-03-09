import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer

def swap_shift_scale(tensor: torch.Tensor) -> torch.Tensor:
    scale, shift = tensor.chunk(2, dim=0) 
    return torch.cat([shift, scale], dim=0)

class DiffusersAdaLN_LoRALayer(LoRALayer):
    '''LoRA layer converted from Diffusers AdaLN, weight is shift-scale swapped'''

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor: 
        # In SD3 and Flux implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
        # while in diffusers it split into scale, shift. 
        # So we swap the linear projection weights in order to be able to use Flux implementation

        weight = super().get_weight(orig_weight) 
        return swap_shift_scale(weight)