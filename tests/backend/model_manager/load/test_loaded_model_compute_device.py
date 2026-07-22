"""Regression tests for issue #9373.

When partial loading is active and VRAM pressure has temporarily offloaded *all* of a VAE's weights back to
RAM, `get_effective_device(model)` reports CPU (it only inspects current parameter residency). The VAE decode
invocations used to move the latents to that inferred device, so the whole decode silently ran on the CPU even
though the model's intended compute device was CUDA.

The fix is to place the latents on the model's *intended* compute device, exposed via
`LoadedModel.compute_device` (which is stable regardless of partial-load residency and still respects cpu_only).
These tests pin down both the plumbing and the underlying mechanism.
"""

import torch

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from tests.backend.model_manager.load.model_cache.cached_model.utils import (
    parameterize_mps_and_cuda,
)


class _ConvModule(torch.nn.Module):
    """A tiny conv stack mirroring how a VAE decoder consumes latents.

    Unlike the shared DummyModule, this has no non-persistent buffers, so offloading really moves every tensor to
    the CPU — which is the residency state that triggers #9373.
    """

    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.conv_out = torch.nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(torch.relu(self.conv_in(x)))


def _make_loaded_model(compute_device: str) -> tuple[LoadedModelWithoutConfig, CachedModelWithPartialLoad]:
    model = _ConvModule()
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(
        model=model, compute_device=torch.device(compute_device), keep_ram_copy=True
    )
    record = CacheRecord(key="test", cached_model=cached_model)
    # The compute_device property only reads through the cache record, so the cache itself is not needed here.
    loaded_model = LoadedModelWithoutConfig(cache_record=record, cache=None)  # type: ignore[arg-type]
    return loaded_model, cached_model


def test_loaded_model_compute_device_reflects_intended_device():
    """LoadedModel.compute_device surfaces the cached model's intended compute device (respects cpu_only)."""
    loaded_model, _ = _make_loaded_model("cpu")
    assert loaded_model.compute_device == torch.device("cpu")


@parameterize_mps_and_cuda
def test_compute_device_stable_when_all_weights_offloaded(device: str):
    """Reproduces #9373: with every weight offloaded to RAM, effective-device inference falls back to CPU, but the
    intended compute device (and thus where the decode should run) stays on the accelerator."""
    loaded_model, cached_model = _make_loaded_model(device)

    # Simulate VRAM pressure: fully load, then push everything back to RAM (as the cache does when working memory
    # leaves no room for the model's weights).
    cached_model.full_load_to_vram()
    cached_model.full_unload_from_vram()
    assert all(p.device.type == "cpu" for p in cached_model.model.parameters())

    # The trap the invocations used to fall into: inferring the device from residency yields CPU...
    assert get_effective_device(cached_model.model) == torch.device("cpu")
    # ...while the intended compute device is preserved. This is what the decode invocations now use.
    assert loaded_model.compute_device == torch.device(device)

    # And it matters: the custom autocast layers cast their weights to the *input's* device, so the input device
    # dictates where the whole forward runs. Placing the input on compute_device keeps the decode on the accelerator.
    x = torch.randn(1, 4, 8, 8)
    out_on_compute_device = cached_model.model(x.to(loaded_model.compute_device))
    assert out_on_compute_device.device.type == device

    # Conversely, following the (wrong) effective device runs the forward entirely on the CPU — the bug.
    out_on_effective_device = cached_model.model(x.to(get_effective_device(cached_model.model)))
    assert out_on_effective_device.device.type == "cpu"
