import torch

from invokeai.backend.lora.conversions.flux_diffusers_lora_conversion_utils import (
    lora_model_from_flux_diffusers_state_dict,
)
from tests.backend.lora.conversions.lora_state_dicts.flux_lora_diffusers_format import state_dict_keys


def test_lora_model_from_flux_diffusers_state_dict():
    """Test that lora_model_from_flux_diffusers_state_dict() can load a state dict in the Diffusers FLUX LoRA format."""
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict: dict[str, torch.Tensor] = {}
    for k in state_dict_keys:
        state_dict[k] = torch.empty(1)

    # Load the state dict into a LoRAModelRaw object.
    model = lora_model_from_flux_diffusers_state_dict(state_dict)

    # Check that the model has the correct number of LoRA layers.
    expected_lora_layers: set[str] = set()
    for k in state_dict_keys:
        k = k.replace("lora_A.weight", "")
        k = k.replace("lora_B.weight", "")
        expected_lora_layers.add(k)
    # Drop the K/V/proj_mlp weights because these are all concatenated into a single layer in the BFL format (we keep
    # the Q weights so that we count these layers once).
    concatenated_weights = ["to_k", "to_v", "proj_mlp", "add_k_proj", "add_v_proj"]
    expected_lora_layers = {k for k in expected_lora_layers if not any(w in k for w in concatenated_weights)}
    assert len(model.layers) == len(expected_lora_layers)
