import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)


class ModelWithRequiredScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.scale = torch.nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            torch.device("cuda"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS device"),
        ),
    ],
)
@pytest.mark.parametrize("keep_ram_copy", [True, False])
@torch.no_grad()
def test_repair_required_tensors_on_compute_device(device: torch.device, keep_ram_copy: bool):
    model = ModelWithRequiredScale()
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=device, keep_ram_copy=keep_ram_copy)

    cached_model._cur_vram_bytes = 0
    repaired_tensors = cached_model.repair_required_tensors_on_compute_device()

    assert repaired_tensors == 1
    assert cached_model._cur_vram_bytes is None
    assert model.scale.device.type == device.type
    assert all(param.device.type == "cpu" for param in model.linear.parameters())
