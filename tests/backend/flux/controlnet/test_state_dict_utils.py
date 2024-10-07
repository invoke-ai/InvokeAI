import pytest
import torch

from invokeai.backend.flux.controlnet.state_dict_utils import (
    convert_diffusers_instantx_state_dict_to_bfl_format,
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


def test_convert_diffusers_instantx_state_dict_to_bfl_format():
    """Smoke test convert_diffusers_instantx_state_dict_to_bfl_format() to ensure that it handles all of the keys."""
    sd = {k: torch.zeros(1) for k in instantx_state_dict_keys}
    bfl_sd = convert_diffusers_instantx_state_dict_to_bfl_format(sd)
    assert bfl_sd is not None
