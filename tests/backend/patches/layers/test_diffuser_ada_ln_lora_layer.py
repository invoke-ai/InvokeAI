import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.diffusers_ada_ln_lora_layer import DiffusersAdaLN_LoRALayer, swap_shift_scale

def test_swap_shift_scale_for_tensor():
    """Test swaping function"""
    tensor = torch.Tensor([1, 2])
    expected = torch.Tensor([2, 1])

    swapped = swap_shift_scale(tensor)
    assert(torch.allclose(expected, swapped))

    size= (3, 4)
    first = torch.randn(size)
    second = torch.randn(size)

    tensor = torch.concat([first, second])
    expected = torch.concat([second, first])

    swapped = swap_shift_scale(tensor)
    assert(torch.allclose(expected, swapped))

def test_diffusers_adaLN_lora_layer_get_weight():
    """Test getting weight from DiffusersAdaLN_LoRALayer."""
    small_in_features = 4
    big_in_features = 8
    out_features = 16
    rank = 4
    alpha = 16.0

    lora = LoRALayer(
        up=torch.ones(out_features, rank), 
        mid=None, 
        down=torch.ones(rank, big_in_features), 
        alpha=alpha, 
        bias=None
    )
    layer = DiffusersAdaLN_LoRALayer(
        up=torch.ones(out_features, rank), 
        mid=None, 
        down=torch.ones(rank, big_in_features), 
        alpha=alpha, 
        bias=None
    )

    # mock original weight, normally ignored in our loRA
    orig_weight = torch.ones(small_in_features)

    diffuser_weight = layer.get_weight(orig_weight)
    lora_weight = lora.get_weight(orig_weight)

    # diffusers lora weight should be flipped
    assert(torch.allclose(diffuser_weight, swap_shift_scale(lora_weight)))

