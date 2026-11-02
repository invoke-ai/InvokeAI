"""Anima denoise wiring: the dequant transient and the sidecar decision.

`peak_dequant_transient_bytes` and `requires_sidecar_patching` are unit-tested next to their module. Pinned here
is the wire: before GGUF support the node passed neither, so a GGUF transformer ran its per-forward dequant copies
outside any reservation, and a LoRA on an int8_tensorwise build -- a `checkpoint` by format -- went through direct
patching, which an `Int8ConvrotLinear` has no `weight` parameter for.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.fields import AnimaConditioningField
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo, ConditioningFieldData

_MODULE = "invokeai.app.invocations.anima.anima_denoise"


class _Stop(Exception):
    """Raised out of the node once the value under test has been handed over."""


def _transformer(linear: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([linear])
    return model


def _gguf_linear(out_features: int = 128, in_features: int = 64) -> torch.nn.Linear:
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    raw = gguf.quantize(np.zeros((out_features, in_features), np.float32), gguf.GGMLQuantizationType.Q8_0)
    weight = GGMLTensor(
        torch.from_numpy(raw), gguf.GGMLQuantizationType.Q8_0, torch.Size((out_features, in_features)), torch.float32
    )
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    return linear


def _int8_linear() -> Int8ConvrotLinear:
    return Int8ConvrotLinear(torch.zeros(64, CONVROT_GROUP_SIZE, dtype=torch.int8), torch.ones(64, 1), convrot=False)


def _run(monkeypatch, transformer: torch.nn.Module, model_format: ModelFormat, on_device) -> MagicMock:
    monkeypatch.setattr(f"{_MODULE}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{_MODULE}.TorchDevice.choose_anima_inference_dtype", lambda _device: torch.float32)
    transformer_info = MagicMock(model=transformer, config=SimpleNamespace(format=model_format))
    transformer_info.model_on_device = MagicMock(side_effect=on_device)
    context = MagicMock()
    context.models.load.return_value = transformer_info
    context.conditioning.load.return_value = ConditioningFieldData(
        conditionings=[AnimaConditioningInfo(qwen3_embeds=torch.zeros(3, 1024), t5xxl_ids=torch.zeros(3).long())]
    )
    invocation = AnimaDenoiseInvocation.model_construct(
        latents=None, noise=None, denoise_mask=None, denoising_start=0.0, denoising_end=1.0, add_noise=True,
        transformer=SimpleNamespace(transformer=object(), loras=[]),
        positive_conditioning=AnimaConditioningField(conditioning_name="pos"), negative_conditioning=None,
        guidance_scale=1.0, width=64, height=64, steps=2, seed=0, control_lllite=None, scheduler="euler",
    )  # fmt: skip
    with pytest.raises(_Stop):
        invocation._run_diffusion(context)
    return transformer_info.model_on_device


def test_a_gguf_transformer_reserves_its_dequant_transient(monkeypatch) -> None:
    """At least the float32 copy of the weight it dequantizes on every forward. The node has no activation
    estimate, so this is the whole request, and the cache floors it at `device_working_mem_gb`."""
    on_device = _run(monkeypatch, _transformer(_gguf_linear()), ModelFormat.GGUFQuantized, _Stop)

    assert on_device.call_args.kwargs["working_mem_bytes"] >= 128 * 64 * torch.float32.itemsize


def test_a_dense_transformer_asks_for_nothing_extra(monkeypatch) -> None:
    on_device = _run(monkeypatch, _transformer(torch.nn.Linear(64, 128)), ModelFormat.Checkpoint, _Stop)

    assert on_device.call_args.kwargs["working_mem_bytes"] == 0


@pytest.mark.parametrize(
    ("linear", "model_format", "sidecar"),
    [
        (_gguf_linear, ModelFormat.GGUFQuantized, True),
        # A `checkpoint` by format; only the loaded module tree shows it is quantized.
        (_int8_linear, ModelFormat.Checkpoint, True),
        (lambda: torch.nn.Linear(64, 128), ModelFormat.Checkpoint, False),
    ],
    ids=["gguf", "int8_tensorwise", "dense"],
)
def test_loras_go_on_as_a_sidecar_exactly_where_the_weights_are_packed(monkeypatch, linear, model_format, sidecar):
    transformer = _transformer(linear())
    captured: dict[str, bool] = {}

    @contextmanager
    def on_device(**_kwargs):
        yield (None, transformer)

    def apply_patches(**kwargs):
        captured["force_sidecar_patching"] = kwargs["force_sidecar_patching"]
        raise _Stop

    monkeypatch.setattr(f"{_MODULE}.LayerPatcher.apply_smart_model_patches", apply_patches)
    _run(monkeypatch, transformer, model_format, on_device)

    assert captured["force_sidecar_patching"] is sidecar
