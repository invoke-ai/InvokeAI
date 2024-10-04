import pytest

from invokeai.backend.flux.controlnet.state_dict_utils import (
    is_state_dict_instantx_controlnet,
    is_state_dict_xlabs_controlnet,
)
from tests.backend.flux.controlnet.instantx_flux_controlnet_state_dict import instantx_state_dict_keys
from tests.backend.flux.controlnet.xlabs_flux_controlnet_state_dict import xlabs_state_dict_keys


@pytest.mark.parametrize(
    ["sd_keys", "expected"],
    [
        (xlabs_state_dict_keys, True),
        (instantx_state_dict_keys, False),
        (["foo"], False),
    ],
)
def test_is_state_dict_xlabs_controlnet(sd_keys: list[str], expected: bool):
    sd = {k: None for k in sd_keys}
    assert is_state_dict_xlabs_controlnet(sd) == expected


@pytest.mark.parametrize(
    ["sd_keys", "expected"],
    [
        (instantx_state_dict_keys, True),
        (xlabs_state_dict_keys, False),
        (["foo"], False),
    ],
)
def test_is_state_dict_instantx_controlnet(sd_keys: list[str], expected: bool):
    sd = {k: None for k in sd_keys}
    assert is_state_dict_instantx_controlnet(sd) == expected
