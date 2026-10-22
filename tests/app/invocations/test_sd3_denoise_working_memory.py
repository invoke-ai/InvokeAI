"""The SD 3 denoise reserves the transient a GGUF transformer dequantizes per forward.

`peak_dequant_transient_bytes` is unit-tested next to its module. What is pinned here is the wire: the node has no
activation estimate, so the dequantization copy is the whole request, and dropping the argument would go unnoticed
until `device_working_mem_gb` is lowered.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch

from invokeai.app.invocations.sd3.sd3_denoise import SD3DenoiseInvocation
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.util.devices import TorchDevice


class _StopBeforeLoad(Exception):
    """Raised out of the mocked `model_on_device()` so the node stops at the reservation."""


def test_a_gguf_transformer_reserves_its_dequant_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(lambda *_args: torch.float32))
    linear = torch.nn.Linear(64, 128, bias=False)
    raw = gguf.quantize(np.zeros((128, 64), np.float32), gguf.GGMLQuantizationType.Q8_0)
    linear.weight = torch.nn.Parameter(
        GGMLTensor(torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, torch.Size((128, 64)), torch.float32),
        requires_grad=False,
    )
    transformer = torch.nn.Sequential(linear)
    transformer.config = SimpleNamespace(joint_attention_dim=64, in_channels=16)
    transformer_info = MagicMock(model=transformer, model_on_device=MagicMock(side_effect=_StopBeforeLoad))
    context = MagicMock()
    context.models.load.return_value = transformer_info
    monkeypatch.setattr(
        SD3DenoiseInvocation,
        "_load_text_conditioning",
        lambda *_args, **_kwargs: (torch.zeros(1, 2, 64), torch.zeros(1, 32)),
    )
    monkeypatch.setattr(SD3DenoiseInvocation, "_prepare_noise_tensor", lambda *_args: torch.zeros(1, 16, 2, 2))
    invocation = SD3DenoiseInvocation.model_construct(
        transformer=SimpleNamespace(transformer=SimpleNamespace()),
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=SimpleNamespace(conditioning_name="neg"),
        latents=None,
        noise=None,
        denoise_mask=None,
        denoising_start=0.0,
        denoising_end=1.0,
        cfg_scale=3.5,
        steps=2,
    )

    with pytest.raises(_StopBeforeLoad):
        invocation.invoke(context)

    # At least the float32 copy of the weight it dequantizes on every forward.
    assert transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"] > 128 * 64 * 4
