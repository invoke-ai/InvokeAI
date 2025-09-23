import accelerate
import pytest

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import get_flux_transformers_params
from invokeai.backend.model_manager.taxonomy import ModelVariantType
from invokeai.backend.patches.lora_conversions.flux_aitoolkit_lora_conversion_utils import (
    _group_state_by_submodel,
    is_state_dict_likely_in_flux_aitoolkit_format,
    lora_model_from_flux_aitoolkit_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_dora_onetrainer_format import (
    state_dict_keys as flux_onetrainer_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_aitoolkit_format import (
    state_dict_keys as flux_aitoolkit_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_is_state_dict_likely_in_flux_aitoolkit_format():
    state_dict = keys_to_mock_state_dict(flux_aitoolkit_state_dict_keys)
    assert is_state_dict_likely_in_flux_aitoolkit_format(state_dict)


@pytest.mark.parametrize("sd_keys", [flux_diffusers_state_dict_keys, flux_onetrainer_state_dict_keys])
def test_is_state_dict_likely_in_flux_kohya_format_false(sd_keys: dict[str, list[int]]):
    state_dict = keys_to_mock_state_dict(sd_keys)
    assert not is_state_dict_likely_in_flux_aitoolkit_format(state_dict)


def test_flux_aitoolkit_transformer_state_dict_is_in_invoke_format():
    state_dict = keys_to_mock_state_dict(flux_aitoolkit_state_dict_keys)
    converted_state_dict = _group_state_by_submodel(state_dict).transformer

    # Extract the prefixes from the converted state dict (without the lora suffixes)
    converted_key_prefixes: list[str] = []
    for k in converted_state_dict.keys():
        k = k.replace(".lora_A.weight", "")
        k = k.replace(".lora_B.weight", "")
        converted_key_prefixes.append(k)

    # Initialize a FLUX model on the meta device.
    with accelerate.init_empty_weights():
        model = Flux(get_flux_transformers_params(ModelVariantType.FluxSchnell))
    model_keys = set(model.state_dict().keys())

    for converted_key_prefix in converted_key_prefixes:
        assert any(model_key.startswith(converted_key_prefix) for model_key in model_keys), (
            f"'{converted_key_prefix}' did not match any model keys."
        )


def test_lora_model_from_flux_aitoolkit_state_dict():
    state_dict = keys_to_mock_state_dict(flux_aitoolkit_state_dict_keys)

    assert lora_model_from_flux_aitoolkit_state_dict(state_dict)
