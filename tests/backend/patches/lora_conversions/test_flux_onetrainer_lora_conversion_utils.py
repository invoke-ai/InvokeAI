import pytest

from invokeai.backend.patches.lora_conversions.flux_onetrainer_lora_conversion_utils import (
    is_state_dict_likely_in_flux_onetrainer_format,
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
