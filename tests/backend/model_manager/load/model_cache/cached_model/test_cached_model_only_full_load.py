import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from tests.backend.model_manager.load.model_cache.dummy_module import DummyModule

parameterize_mps_and_cuda = pytest.mark.parametrize(
    ("device"),
    [
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available.")
        ),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
    ],
)


@parameterize_mps_and_cuda
def test_cached_model_total_bytes(device: str):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(model=model, compute_device=torch.device(device), total_bytes=100)
    assert cached_model.total_bytes() == 100


@parameterize_mps_and_cuda
def test_cached_model_is_in_vram(device: str):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(model=model, compute_device=torch.device(device), total_bytes=100)
    assert not cached_model.is_in_vram()

    cached_model.full_load_to_vram()
    assert cached_model.is_in_vram()

    cached_model.full_unload_from_vram()
    assert not cached_model.is_in_vram()


@parameterize_mps_and_cuda
def test_cached_model_full_load_and_unload(device: str):
    model = DummyModule()
    cached_model = CachedModelOnlyFullLoad(model=model, compute_device=torch.device(device), total_bytes=100)
    assert cached_model.full_load_to_vram() == 100
    assert cached_model.is_in_vram()
    assert all(p.device.type == device for p in cached_model.model.parameters())

    assert cached_model.full_unload_from_vram() == 100
    assert not cached_model.is_in_vram()
    assert all(p.device.type == "cpu" for p in cached_model.model.parameters())
