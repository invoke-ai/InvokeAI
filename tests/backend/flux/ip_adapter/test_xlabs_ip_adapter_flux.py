import sys

import accelerate
import pytest
import torch

from invokeai.backend.flux.ip_adapter.state_dict_utils import (
    infer_xlabs_ip_adapter_params_from_state_dict,
    is_state_dict_xlabs_ip_adapter,
)
from invokeai.backend.flux.ip_adapter.xlabs_ip_adapter_flux import (
    XlabsIpAdapterFlux,
    XlabsIpAdapterParams,
)
from tests.backend.flux.ip_adapter.xlabs_flux_ip_adapter_state_dict import xlabs_flux_ip_adapter_sd_shapes
from tests.backend.flux.ip_adapter.xlabs_flux_ip_adapter_v2_state_dict import xlabs_flux_ip_adapter_v2_sd_shapes


@pytest.mark.parametrize("sd_shapes", [xlabs_flux_ip_adapter_sd_shapes, xlabs_flux_ip_adapter_v2_sd_shapes])
def test_is_state_dict_xlabs_ip_adapter(sd_shapes: dict[str, list[int]]):
    # Construct a dummy state_dict.
    sd = {k: None for k in sd_shapes}

    assert is_state_dict_xlabs_ip_adapter(sd)


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
@pytest.mark.parametrize(
    ["sd_shapes", "expected_params"],
    [
        (
            xlabs_flux_ip_adapter_sd_shapes,
            XlabsIpAdapterParams(
                num_double_blocks=19,
                context_dim=4096,
                hidden_dim=3072,
                clip_embeddings_dim=768,
                clip_extra_context_tokens=4,
            ),
        ),
        (
            xlabs_flux_ip_adapter_v2_sd_shapes,
            XlabsIpAdapterParams(
                num_double_blocks=19,
                context_dim=4096,
                hidden_dim=3072,
                clip_embeddings_dim=768,
                clip_extra_context_tokens=16,
            ),
        ),
    ],
)
def test_infer_xlabs_ip_adapter_params_from_state_dict(
    sd_shapes: dict[str, list[int]], expected_params: XlabsIpAdapterParams
):
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in sd_shapes.items()}

    params = infer_xlabs_ip_adapter_params_from_state_dict(sd)
    assert params == expected_params


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
@pytest.mark.parametrize("sd_shapes", [xlabs_flux_ip_adapter_sd_shapes, xlabs_flux_ip_adapter_v2_sd_shapes])
def test_initialize_xlabs_ip_adapter_flux_from_state_dict(sd_shapes: dict[str, list[int]]):
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in sd_shapes.items()}

    # Initialize the XLabs IP-Adapter from the state_dict.
    params = infer_xlabs_ip_adapter_params_from_state_dict(sd)

    with accelerate.init_empty_weights():
        model = XlabsIpAdapterFlux(params=params)

    # Smoke test state_dict loading.
    model.load_xlabs_state_dict(sd)
