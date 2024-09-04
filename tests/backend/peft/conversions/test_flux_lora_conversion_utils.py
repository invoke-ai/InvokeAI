import pytest
import torch

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import params
from invokeai.backend.peft.conversions.flux_lora_conversion_utils import (
    convert_flux_kohya_state_dict_to_invoke_format,
    is_state_dict_likely_in_flux_kohya_format,
)
from tests.backend.peft.conversions.lora_state_dicts.flux_lora_kohya_format import state_dict_keys


def test_is_state_dict_likely_in_flux_kohya_format_true():
    """Test that is_state_dict_likely_in_flux_kohya_format() can identify a state dict in the Kohya FLUX LoRA format."""
    # Construct a state dict that is in the Kohya FLUX LoRA format.
    state_dict: dict[str, torch.Tensor] = {}
    for k in state_dict_keys:
        state_dict[k] = torch.empty(1)
    assert is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_is_state_dict_likely_in_flux_kohya_format_false():
    """Test that is_state_dict_likely_in_flux_kohya_format() returns False for a state dict that is not in the Kohya FLUX LoRA format."""
    state_dict: dict[str, torch.Tensor] = {
        "unexpected_key.lora_up.weight": torch.empty(1),
    }
    assert not is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_convert_flux_kohya_state_dict_to_invoke_format():
    # Construct state_dict from state_dict_keys.
    state_dict: dict[str, torch.Tensor] = {}
    for k in state_dict_keys:
        state_dict[k] = torch.empty(1)

    converted_state_dict = convert_flux_kohya_state_dict_to_invoke_format(state_dict)

    # Extract the prefixes from the converted state dict (i.e. without the .lora_up.weight, .lora_down.weight, and
    # .alpha suffixes).
    converted_key_prefixes: list[str] = []
    for k in converted_state_dict.keys():
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        converted_key_prefixes.append(k)

    # Initialize a FLUX model on the meta device.
    with torch.device("meta"):
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


def test_convert_flux_kohya_state_dict_to_invoke_format_error():
    """Test that an error is raised by convert_flux_kohya_state_dict_to_invoke_format() if the input state_dict contains
    unexpected keys.
    """
    state_dict = {
        "unexpected_key.lora_up.weight": torch.empty(1),
    }

    with pytest.raises(ValueError):
        convert_flux_kohya_state_dict_to_invoke_format(state_dict)
