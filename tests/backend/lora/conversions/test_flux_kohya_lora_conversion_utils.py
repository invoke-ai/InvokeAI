import accelerate
import pytest
import torch

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import params
from invokeai.backend.lora.conversions.flux_kohya_lora_conversion_utils import (
    _convert_flux_transformer_kohya_state_dict_to_invoke_format,
    is_state_dict_likely_in_flux_kohya_format,
    lora_model_from_flux_kohya_state_dict,
)
from tests.backend.lora.conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.lora.conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.lora.conversions.lora_state_dicts.flux_lora_kohya_with_te1_format import (
    state_dict_keys as flux_kohya_te1_state_dict_keys,
)
from tests.backend.lora.conversions.lora_state_dicts.utils import keys_to_mock_state_dict


@pytest.mark.parametrize("sd_keys", [flux_kohya_state_dict_keys, flux_kohya_te1_state_dict_keys])
def test_is_state_dict_likely_in_flux_kohya_format_true(sd_keys: list[str]):
    """Test that is_state_dict_likely_in_flux_kohya_format() can identify a state dict in the Kohya FLUX LoRA format."""
    # Construct a state dict that is in the Kohya FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)

    assert is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_is_state_dict_likely_in_flux_kohya_format_false():
    """Test that is_state_dict_likely_in_flux_kohya_format() returns False for a state dict that is in the Diffusers
    FLUX LoRA format.
    """
    state_dict = keys_to_mock_state_dict(flux_diffusers_state_dict_keys)
    assert not is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_convert_flux_transformer_kohya_state_dict_to_invoke_format():
    # Construct state_dict from state_dict_keys.
    state_dict = keys_to_mock_state_dict(flux_kohya_state_dict_keys)

    converted_state_dict = _convert_flux_transformer_kohya_state_dict_to_invoke_format(state_dict)

    # Extract the prefixes from the converted state dict (i.e. without the .lora_up.weight, .lora_down.weight, and
    # .alpha suffixes).
    converted_key_prefixes: list[str] = []
    for k in converted_state_dict.keys():
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        converted_key_prefixes.append(k)

    # Initialize a FLUX model on the meta device.
    with accelerate.init_empty_weights():
        model = Flux(params["flux-dev"])
    model_keys = set(model.state_dict().keys())

    # Assert that the converted state dict matches the keys in the actual model.
    for converted_key_prefix in converted_key_prefixes:
        found_match = False
        for model_key in model_keys:
            if model_key.startswith(converted_key_prefix):
                found_match = True
                break
        if not found_match:
            raise AssertionError(f"Could not find a match for the converted key prefix: {converted_key_prefix}")


def test_convert_flux_transformer_kohya_state_dict_to_invoke_format_error():
    """Test that an error is raised by _convert_flux_transformer_kohya_state_dict_to_invoke_format() if the input
    state_dict contains unexpected keys.
    """
    state_dict = {
        "unexpected_key.lora_up.weight": torch.empty(1),
    }

    with pytest.raises(ValueError):
        _convert_flux_transformer_kohya_state_dict_to_invoke_format(state_dict)


@pytest.mark.parametrize("sd_keys", [flux_kohya_state_dict_keys, flux_kohya_te1_state_dict_keys])
def test_lora_model_from_flux_kohya_state_dict(sd_keys: list[str]):
    """Test that a LoRAModelRaw can be created from a state dict in the Kohya FLUX LoRA format."""
    # Construct a state dict that is in the Kohya FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)

    lora_model = lora_model_from_flux_kohya_state_dict(state_dict)

    # Prepare expected layer keys.
    expected_layer_keys: set[str] = set()
    for k in sd_keys:
        # Remove prefixes.
        k = k.replace("lora_unet_", "")
        k = k.replace("lora_te1_", "")
        # Remove suffixes.
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        expected_layer_keys.add(k)

    # Assert that the lora_model has the expected layers.
    lora_model_keys = set(lora_model.layers.keys())
    lora_model_keys = {k.replace(".", "_") for k in lora_model_keys}
    assert lora_model_keys == expected_layer_keys
