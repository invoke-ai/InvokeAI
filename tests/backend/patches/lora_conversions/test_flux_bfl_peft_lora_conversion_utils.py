import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    is_state_dict_likely_in_flux_bfl_peft_format,
    lora_model_from_flux2_bfl_peft_state_dict,
    lora_model_from_flux_bfl_peft_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX


@pytest.mark.parametrize("magnitude_suffix", ["magnitude", "lora_magnitude_vector.weight"])
def test_dora_magnitude_is_preserved(magnitude_suffix: str):
    in_dim, out_dim, rank = 5, 7, 2
    magnitude = torch.arange(1, out_dim + 1, dtype=torch.float32)
    prefix = "diffusion_model.double_blocks.0.img_attn.proj"
    state_dict = {
        f"{prefix}.lora_A.weight": torch.zeros(rank, in_dim),
        f"{prefix}.lora_B.weight": torch.zeros(out_dim, rank),
        f"{prefix}.{magnitude_suffix}": magnitude,
    }

    assert is_state_dict_likely_in_flux_bfl_peft_format(state_dict)
    model = lora_model_from_flux_bfl_peft_state_dict(state_dict)
    layer = model.layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.proj"]

    assert isinstance(layer, DoRALayer)
    assert layer.magnitude_is_out_dim is True
    assert torch.equal(layer.dora_scale, magnitude)
    assert layer.get_weight(torch.randn(out_dim, in_dim)).shape == (out_dim, in_dim)


def test_flux2_fused_qkv_dora_magnitude_is_split_by_output():
    hidden_dim, rank = 6, 2
    magnitude = torch.arange(1, 3 * hidden_dim + 1, dtype=torch.float32)
    prefix = "diffusion_model.double_blocks.0.img_attn.qkv"
    state_dict = {
        f"{prefix}.lora_A.weight": torch.zeros(rank, hidden_dim),
        f"{prefix}.lora_B.weight": torch.zeros(3 * hidden_dim, rank),
        f"{prefix}.magnitude": magnitude,
    }

    model = lora_model_from_flux2_bfl_peft_state_dict(state_dict)
    layer_names = ("to_q", "to_k", "to_v")
    for index, layer_name in enumerate(layer_names):
        key = f"{FLUX_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.{layer_name}"
        layer = model.layers[key]
        expected_magnitude = magnitude[index * hidden_dim : (index + 1) * hidden_dim]

        assert isinstance(layer, DoRALayer)
        assert layer.magnitude_is_out_dim is True
        assert torch.equal(layer.dora_scale, expected_magnitude)
        assert layer.get_weight(torch.randn(hidden_dim, hidden_dim)).shape == (hidden_dim, hidden_dim)
