import sys

import pytest
import torch

from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.state_dict_utils import (
    convert_diffusers_instantx_state_dict_to_bfl_format,
    infer_flux_params_from_state_dict,
    infer_instantx_num_control_modes_from_state_dict,
    is_state_dict_instantx_controlnet,
    is_state_dict_xlabs_controlnet,
)
from tests.backend.flux.controlnet.instantx_flux_controlnet_state_dict import instantx_config, instantx_sd_shapes
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


# TODO(ryand): Figure out why some tests in this file are failing on the MacOS CI runners. It seems to be related to
# using the meta device. I can't reproduce the issue on my local MacOS system.


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
def test_infer_flux_params_from_state_dict():
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in instantx_sd_shapes.items()}

    sd = convert_diffusers_instantx_state_dict_to_bfl_format(sd)
    flux_params = infer_flux_params_from_state_dict(sd)

    assert flux_params.in_channels == instantx_config["in_channels"]
    assert flux_params.vec_in_dim == instantx_config["pooled_projection_dim"]
    assert flux_params.context_in_dim == instantx_config["joint_attention_dim"]
    assert flux_params.hidden_size // flux_params.num_heads == instantx_config["attention_head_dim"]
    assert flux_params.num_heads == instantx_config["num_attention_heads"]
    assert flux_params.mlp_ratio == 4
    assert flux_params.depth == instantx_config["num_layers"]
    assert flux_params.depth_single_blocks == instantx_config["num_single_layers"]
    assert flux_params.axes_dim == instantx_config["axes_dims_rope"]
    assert flux_params.theta == 10000
    assert flux_params.qkv_bias
    assert flux_params.guidance_embed == instantx_config["guidance_embeds"]


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
def test_infer_instantx_num_control_modes_from_state_dict():
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in instantx_sd_shapes.items()}

    sd = convert_diffusers_instantx_state_dict_to_bfl_format(sd)
    num_control_modes = infer_instantx_num_control_modes_from_state_dict(sd)

    assert num_control_modes == instantx_config["num_mode"]


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
def test_load_instantx_from_state_dict():
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in instantx_sd_shapes.items()}

    sd = convert_diffusers_instantx_state_dict_to_bfl_format(sd)
    flux_params = infer_flux_params_from_state_dict(sd)
    num_control_modes = infer_instantx_num_control_modes_from_state_dict(sd)

    with torch.device("meta"):
        model = InstantXControlNetFlux(flux_params, num_control_modes)

    model_sd = model.state_dict()

    assert set(model_sd.keys()) == set(sd.keys())
    for key, tensor in model_sd.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == sd[key].shape
