import pytest
import torch

from invokeai.backend.flux.controlnet.state_dict_utils import (
    convert_diffusers_instantx_state_dict_to_bfl_format,
    is_state_dict_instantx_controlnet,
    is_state_dict_xlabs_controlnet,
)
from tests.backend.flux.controlnet.instantx_flux_controlnet_state_dict import instantx_sd_shapes
from tests.backend.flux.controlnet.xlabs_flux_controlnet_state_dict import xlabs_sd_shapes


@pytest.mark.parametrize(
    ["sd_shapes", "expected"],
    [
        (xlabs_sd_shapes, True),
        (instantx_sd_shapes, False),
        (["foo"], False),
    ],
)
def test_is_state_dict_xlabs_controlnet(sd_shapes: dict[str, list[int]], expected: bool):
    sd = {k: None for k in sd_shapes}
    assert is_state_dict_xlabs_controlnet(sd) == expected


@pytest.mark.parametrize(
    ["sd_keys", "expected"],
    [
        (instantx_sd_shapes, True),
        (xlabs_sd_shapes, False),
        (["foo"], False),
    ],
)
def test_is_state_dict_instantx_controlnet(sd_keys: list[str], expected: bool):
    sd = {k: None for k in sd_keys}
    assert is_state_dict_instantx_controlnet(sd) == expected


def test_convert_diffusers_instantx_state_dict_to_bfl_format():
    """Smoke test convert_diffusers_instantx_state_dict_to_bfl_format() to ensure that it handles all of the keys."""
    sd = {k: torch.zeros(1) for k in instantx_sd_shapes}
    bfl_sd = convert_diffusers_instantx_state_dict_to_bfl_format(sd)
    assert bfl_sd is not None
