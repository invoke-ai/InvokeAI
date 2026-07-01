# The state dict keys and shapes for a FLUX Redux model.
# Model source: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/blob/1282f955f706b5240161278f2ef261d2a29ad649/flux1-redux-dev.safetensors
# The keys and shapes were extracted with extract_sd_keys_and_shapes.py.
import torch

from invokeai.backend.flux.redux.flux_redux_state_dict_utils import is_state_dict_likely_flux_redux
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict

flux_redux_keys_and_shapes = {
    "redux_down.bias": [4096],
    "redux_down.weight": [4096, 12288],
    "redux_up.bias": [12288],
    "redux_up.weight": [12288, 1152],
}


def test_is_state_dict_likely_flux_redux_true():
    # Expand flux_redux_keys_and_shapes to a mock state dict.
    sd = keys_to_mock_state_dict(flux_redux_keys_and_shapes)
    assert is_state_dict_likely_flux_redux(sd)


def test_is_state_dict_likely_flux_redux_extra_key():
    # Expand flux_redux_keys_and_shapes to a mock state dict.
    sd = keys_to_mock_state_dict(flux_redux_keys_and_shapes)
    # Add an extra key to the state dict.
    sd["extra_key"] = torch.rand(1)
    assert not is_state_dict_likely_flux_redux(sd)
