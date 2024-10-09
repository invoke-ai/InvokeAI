import pytest
import torch

from invokeai.backend.lora.conversions.flux_diffusers_lora_conversion_utils import (
    is_state_dict_likely_in_flux_diffusers_format,
    lora_model_from_flux_diffusers_state_dict,
)
from invokeai.backend.lora.conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from tests.backend.lora.conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.lora.conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.lora.conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_is_state_dict_likely_in_flux_diffusers_format_true():
    """Test that is_state_dict_likely_in_flux_diffusers_format() can identify a state dict in the Diffusers FLUX LoRA format."""
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_diffusers_state_dict_keys)

    assert is_state_dict_likely_in_flux_diffusers_format(state_dict)


def test_is_state_dict_likely_in_flux_diffusers_format_false():
    """Test that is_state_dict_likely_in_flux_diffusers_format() returns False for a state dict that is not in the Kohya
    FLUX LoRA format.
    """
    # Construct a state dict that is not in the Kohya FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_kohya_state_dict_keys)

    assert not is_state_dict_likely_in_flux_diffusers_format(state_dict)


def test_lora_model_from_flux_diffusers_state_dict():
    """Test that lora_model_from_flux_diffusers_state_dict() can load a state dict in the Diffusers FLUX LoRA format."""
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_diffusers_state_dict_keys)
    # Load the state dict into a LoRAModelRaw object.
    model = lora_model_from_flux_diffusers_state_dict(state_dict, alpha=8.0)

    # Check that the model has the correct number of LoRA layers.
    expected_lora_layers: set[str] = set()
    for k in flux_diffusers_state_dict_keys:
        k = k.replace("lora_A.weight", "")
        k = k.replace("lora_B.weight", "")
        expected_lora_layers.add(k)
    # Drop the K/V/proj_mlp weights because these are all concatenated into a single layer in the BFL format (we keep
    # the Q weights so that we count these layers once).
    concatenated_weights = ["to_k", "to_v", "proj_mlp", "add_k_proj", "add_v_proj"]
    expected_lora_layers = {k for k in expected_lora_layers if not any(w in k for w in concatenated_weights)}
    assert len(model.layers) == len(expected_lora_layers)
    assert all(k.startswith(FLUX_LORA_TRANSFORMER_PREFIX) for k in model.layers.keys())


def test_lora_model_from_flux_diffusers_state_dict_extra_keys_error():
    """Test that lora_model_from_flux_diffusers_state_dict() raises an error if the input state_dict contains unexpected
    keys that we don't handle.
    """
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_diffusers_state_dict_keys)
    # Add an unexpected key.
    state_dict["transformer.single_transformer_blocks.0.unexpected_key.lora_A.weight"] = torch.empty(1)

    # Check that an error is raised.
    with pytest.raises(AssertionError):
        lora_model_from_flux_diffusers_state_dict(state_dict, alpha=8.0)
