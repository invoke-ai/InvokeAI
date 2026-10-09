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
        self.register_buffer("transient", torch.ones(4), persistent=False)

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


def test_repair_failure_invalidates_accounting_and_retry_repairs_remaining_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    cached_model._cur_vram_bytes = 0
    original_load = model[1]._load_from_state_dict

    def fail_second_module(*args, **kwargs):
        raise RuntimeError("second module repair failed")

    monkeypatch.setattr(model[1], "_load_from_state_dict", fail_second_module)

    with pytest.raises(RuntimeError, match="second module repair failed"):
        cached_model.repair_required_tensors_on_compute_device()

    assert model[0].weight.is_meta
    assert not model[1].weight.is_meta
    assert cached_model._cur_vram_bytes is None

    monkeypatch.setattr(model[1], "_load_from_state_dict", original_load)
    repaired_tensors = cached_model.repair_required_tensors_on_compute_device()

    assert repaired_tensors == 2
    assert all(param.is_meta for param in model.parameters())
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()


def test_repair_retries_non_persistent_buffer_after_required_tensor_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ModelWithRequiredScale()
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("meta"), keep_ram_copy=False)
    cached_model._cur_vram_bytes = 0
    move_buffers = cached_model._move_non_persistent_buffers_to_device

    def fail_buffer_move(device: torch.device) -> None:
        raise RuntimeError("simulated buffer repair failure")

    monkeypatch.setattr(cached_model, "_move_non_persistent_buffers_to_device", fail_buffer_move)

    with pytest.raises(RuntimeError, match="simulated buffer repair failure"):
        cached_model.repair_required_tensors_on_compute_device()

    assert model.scale.is_meta
    assert model.transient.device.type == "cpu"
    assert cached_model._cur_vram_bytes is None

    monkeypatch.setattr(cached_model, "_move_non_persistent_buffers_to_device", move_buffers)
    assert cached_model.repair_required_tensors_on_compute_device() == 0
    assert model.transient.is_meta

    repaired_buffer = model.transient
    assert cached_model.repair_required_tensors_on_compute_device() == 0
    assert model.transient is repaired_buffer
