"""`requires_sidecar_patching` for quantized Linears the format does not announce.

Ideogram 4's nf4 pair is a Diffusers folder, so the format says nothing; its bitsandbytes Linears pack the weight into
a uint8 parameter, and a direct patch skips every such layer as a shape mismatch -- the LoRA would silently do
nothing. The helper must find them in the module tree without importing bitsandbytes, which is optional.
"""

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching


def test_a_dense_diffusers_model_is_patched_directly() -> None:
    assert not requires_sidecar_patching(torch.nn.Sequential(torch.nn.Linear(4, 4)), ModelFormat.Diffusers)


def test_an_nf4_model_in_a_diffusers_folder_needs_the_sidecar() -> None:
    pytest.importorskip("bitsandbytes")
    from invokeai.backend.quantization.bnb_nf4 import InvokeLinearNF4

    model = torch.nn.Sequential(torch.nn.Linear(64, 64), InvokeLinearNF4(input_features=64, output_features=64))

    assert requires_sidecar_patching(model, ModelFormat.Diffusers)
