import accelerate
import pytest
import torch

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import get_flux_transformers_params
from invokeai.backend.model_manager.taxonomy import FluxVariantType
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.flux_xlabs_lora_conversion_utils import (
    is_state_dict_likely_in_flux_xlabs_format,
    lora_model_from_flux_xlabs_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_xlabs_format import (
    state_dict_keys as flux_xlabs_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_is_state_dict_likely_in_flux_xlabs_format_true():
    """Test that is_state_dict_likely_in_flux_xlabs_format() can identify a state dict in the xlabs FLUX LoRA format."""
    state_dict = keys_to_mock_state_dict(flux_xlabs_state_dict_keys)
    assert is_state_dict_likely_in_flux_xlabs_format(state_dict)


@pytest.mark.parametrize("sd_keys", [flux_diffusers_state_dict_keys, flux_kohya_state_dict_keys])
def test_is_state_dict_likely_in_flux_xlabs_format_false(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_xlabs_format() returns False for state dicts in other formats."""
    state_dict = keys_to_mock_state_dict(sd_keys)
    assert not is_state_dict_likely_in_flux_xlabs_format(state_dict)


def test_lora_model_from_flux_xlabs_state_dict():
    """Test that a ModelPatchRaw can be created from a state dict in the xlabs FLUX LoRA format."""
    state_dict = keys_to_mock_state_dict(flux_xlabs_state_dict_keys)

    lora_model = lora_model_from_flux_xlabs_state_dict(state_dict)

    # Verify the expected layer keys are created
    expected_layer_keys = {
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.proj",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.img_attn.qkv",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.txt_attn.proj",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.0.txt_attn.qkv",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.1.img_attn.proj",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.1.img_attn.qkv",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.1.txt_attn.proj",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.1.txt_attn.qkv",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.10.img_attn.proj",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.10.img_attn.qkv",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.10.txt_attn.proj",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}double_blocks.10.txt_attn.qkv",
    }

    assert set(lora_model.layers.keys()) == expected_layer_keys


def test_lora_model_from_flux_xlabs_state_dict_matches_model_keys():
    """Test that the converted xlabs LoRA keys match the actual FLUX model keys."""
    state_dict = keys_to_mock_state_dict(flux_xlabs_state_dict_keys)

    lora_model = lora_model_from_flux_xlabs_state_dict(state_dict)

    # Extract the layer prefixes (without the lora_transformer- prefix)
    converted_key_prefixes: list[str] = []
    for k in lora_model.layers.keys():
        # Remove the transformer prefix
        k = k.replace(FLUX_LORA_TRANSFORMER_PREFIX, "")
        converted_key_prefixes.append(k)

    # Initialize a FLUX model on the meta device.
    with accelerate.init_empty_weights():
        model = Flux(get_flux_transformers_params(FluxVariantType.Schnell))
    model_keys = set(model.state_dict().keys())

    # Assert that the converted keys match prefixes in the actual model.
    for converted_key_prefix in converted_key_prefixes:
        found_match = False
        for model_key in model_keys:
            if model_key.startswith(converted_key_prefix):
                found_match = True
                break
        if not found_match:
            raise AssertionError(f"Could not find a match for the converted key prefix: {converted_key_prefix}")


def test_lora_model_from_flux_xlabs_state_dict_error():
    """Test that an error is raised if the input state_dict contains unexpected keys."""
    state_dict = {
        "unexpected_key.down.weight": torch.empty(1),
    }

    with pytest.raises(ValueError):
        lora_model_from_flux_xlabs_state_dict(state_dict)
