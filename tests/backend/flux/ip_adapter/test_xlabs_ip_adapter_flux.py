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
)
from tests.backend.flux.ip_adapter.xlabs_flux_ip_adapter_state_dict import xlabs_sd_shapes


def test_is_state_dict_xlabs_ip_adapter():
    # Construct a dummy state_dict.
    sd = {k: None for k in xlabs_sd_shapes}

    assert is_state_dict_xlabs_ip_adapter(sd)


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
def test_infer_xlabs_ip_adapter_params_from_state_dict():
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in xlabs_sd_shapes.items()}

    params = infer_xlabs_ip_adapter_params_from_state_dict(sd)

    assert params.num_double_blocks == 19
    assert params.context_dim == 4096
    assert params.hidden_dim == 3072
    assert params.clip_embeddings_dim == 768


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
def test_initialize_xlabs_ip_adapter_flux_from_state_dict():
    # Construct a dummy state_dict with tensors of the correct shape on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(v) for k, v in xlabs_sd_shapes.items()}

    # Initialize the XLabs IP-Adapter from the state_dict.
    params = infer_xlabs_ip_adapter_params_from_state_dict(sd)

    with accelerate.init_empty_weights():
        model = XlabsIpAdapterFlux(params=params)

    # Smoke test state_dict loading.
    model.load_xlabs_state_dict(sd)
