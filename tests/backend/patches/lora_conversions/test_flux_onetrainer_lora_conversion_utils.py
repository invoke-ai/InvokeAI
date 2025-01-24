import pytest

from invokeai.backend.patches.lora_conversions.flux_lora_constants import (
    FLUX_LORA_CLIP_PREFIX,
    FLUX_LORA_T5_PREFIX,
    FLUX_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.flux_onetrainer_lora_conversion_utils import (
    is_state_dict_likely_in_flux_onetrainer_format,
    lora_model_from_flux_onetrainer_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_dora_onetrainer_format import (
    state_dict_keys as flux_onetrainer_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_with_te1_format import (
    state_dict_keys as flux_kohya_te1_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_is_state_dict_likely_in_flux_onetrainer_format_true():
    """Test that is_state_dict_likely_in_flux_onetrainer_format() can identify a state dict in the OneTrainer
    FLUX LoRA format.
    """
    # Construct a state dict that is in the OneTrainer FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_onetrainer_state_dict_keys)

    assert is_state_dict_likely_in_flux_onetrainer_format(state_dict)


@pytest.mark.parametrize(
    "sd_keys",
    [
        flux_kohya_state_dict_keys,
        flux_kohya_te1_state_dict_keys,
        flux_diffusers_state_dict_keys,
    ],
)
def test_is_state_dict_likely_in_flux_onetrainer_format_false(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_onetrainer_format() returns False for a state dict that is in the Diffusers
    FLUX LoRA format.
    """
    state_dict = keys_to_mock_state_dict(sd_keys)
    assert not is_state_dict_likely_in_flux_onetrainer_format(state_dict)


def test_lora_model_from_flux_onetrainer_state_dict():
    state_dict = keys_to_mock_state_dict(flux_onetrainer_state_dict_keys)

    lora_model = lora_model_from_flux_onetrainer_state_dict(state_dict)

    # Check that the model has the correct number of LoRA layers.
    expected_lora_layers: set[str] = set()
    for k in flux_onetrainer_state_dict_keys:
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        k = k.replace(".dora_scale", "")
        expected_lora_layers.add(k)
    # Drop the K/V/proj_mlp weights because these are all concatenated into a single layer in the BFL format (we keep
    # the Q weights so that we count these layers once).
    concatenated_weights = ["to_k", "to_v", "proj_mlp", "add_k_proj", "add_v_proj"]
    expected_lora_layers = {k for k in expected_lora_layers if not any(w in k for w in concatenated_weights)}

    assert len(lora_model.layers) == len(expected_lora_layers)

    # Check that all of the layers have the expected prefix.
    assert all(
        k.startswith((FLUX_LORA_TRANSFORMER_PREFIX, FLUX_LORA_CLIP_PREFIX, FLUX_LORA_T5_PREFIX))
        for k in lora_model.layers.keys()
    )
