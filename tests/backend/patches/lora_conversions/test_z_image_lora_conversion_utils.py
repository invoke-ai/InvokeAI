import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.z_image_lora_conversion_utils import (
    lora_model_from_z_image_state_dict,
)


@pytest.mark.parametrize("magnitude_suffix", ["magnitude", "lora_magnitude_vector.weight"])
def test_peft_dora_magnitude_is_preserved(magnitude_suffix: str):
    in_dim, out_dim, rank = 5, 7, 2
    magnitude = torch.arange(1, out_dim + 1, dtype=torch.float32)
    prefix = "diffusion_model.layers.0.attention.to_q"
    state_dict = {
        f"{prefix}.lora_A.weight": torch.zeros(rank, in_dim),
        f"{prefix}.lora_B.weight": torch.zeros(out_dim, rank),
        f"{prefix}.{magnitude_suffix}": magnitude,
    }

    model = lora_model_from_z_image_state_dict(state_dict)
    layer = model.layers[f"{Z_IMAGE_LORA_TRANSFORMER_PREFIX}layers.0.attention.to_q"]

    assert isinstance(layer, DoRALayer)
    assert layer.magnitude_is_out_dim is True
    assert torch.equal(layer.dora_scale, magnitude)
    assert layer.get_weight(torch.randn(out_dim, in_dim)).shape == (out_dim, in_dim)
